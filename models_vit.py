# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import math
import timm.models.vision_transformer
import sys

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, batch_size=1024, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        # self.pos_drop = nn.Dropout(p=0.0)  # 최근 ViT에서는 pos drop을 0.0으로 둠
        max_pruning_ratio = 0.5
        # dynamic subsampling
        codebook_size = 32
        aggregation = 'attention'
        locality_constraint = True
        energy_function = 'local_linear'
        embed_dim = 768
        if codebook_size > 0 and max_pruning_ratio > 0:  # dynamic subsampling
            self.dynamic_subsampling = True
        else:
            self.dynamic_subsampling = False

        self.codebook_size = codebook_size
        self.max_pruning_ratio = max_pruning_ratio
        self.energy_function = energy_function

        # running threshold (EMA)
        self.register_buffer("energy_threshold", torch.tensor(0.0), persistent=True)

        self.state_vectors = torch.nn.Linear(embed_dim, codebook_size)
        self.energy_func1 = torch.nn.Linear(embed_dim, codebook_size)
        self.energy_func2 = torch.nn.Linear(embed_dim, codebook_size)
        self.energy_func3 = torch.nn.Linear(embed_dim, codebook_size)
        self.energy_func4 = torch.nn.Linear(embed_dim, codebook_size)
        self.kl_loss = torch.nn.KLDivLoss(reduction="none")
        self.eps = 1e-6
        self.state_prediction_loss = None
        self.entropy_maximization_loss = None
        self.l2_regularization_loss = None
        self.locality_constraint = locality_constraint
        self.aggregation = aggregation

        if aggregation == 'cnn':
            self.aggregation_cnn1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=True)
            self.aggregation_cnn2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=True)
        elif aggregation == 'attention':
            self.linear_q = torch.nn.Linear(embed_dim, 64)
            self.linear_k = torch.nn.Linear(embed_dim, 64)

            att_context_size = 1
            self.batch_size = batch_size
            x_axis_size = 14  # TODO: change dynamically
            att_mask = torch.ones(self.batch_size, x_axis_size*x_axis_size, x_axis_size*x_axis_size)
            att_mask = att_mask.triu(diagonal=-att_context_size)
            att_mask = att_mask.tril(diagonal=att_context_size)

            att_mask_a = torch.ones(self.batch_size, x_axis_size*x_axis_size, x_axis_size*x_axis_size)
            att_mask_a = att_mask_a.triu(diagonal=-att_context_size*x_axis_size)
            att_mask_a = att_mask_a.tril(diagonal=att_context_size*x_axis_size)

            att_mask_b = torch.ones(self.batch_size, x_axis_size*x_axis_size, x_axis_size*x_axis_size)
            att_mask_b = att_mask_b.triu(diagonal=-x_axis_size)
            att_mask_b = att_mask_b.tril(diagonal=x_axis_size)

            self.att_mask = att_mask_a + att_mask_b + att_mask




        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x, print_option = False):
        print_option = True
        B = x.shape[0]
        if print_option and torch.isnan(x).any():
            print("x is nan before patch_embed")
        x = self.patch_embed(x)
        if print_option and torch.isnan(x).any():
            print("x is nan after patch_embed")
            
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if print_option and torch.isnan(cls_tokens).any():
            print("cls_tokens is nan")
        x = torch.cat((cls_tokens, x), dim=1)
        if print_option and torch.isnan(x).any():
            print("x is nan after cat with cls_tokens")
        # x = x + self.pos_embed

        if print_option and torch.isnan(x).any():
            print("x size before: ", x.size())

    
        # dynamic subsampling
        cls_tok, x_wo_aug = x[:, :1, :].clone().detach(), x[:, 1:, :].clone().detach()
        B, T_wo, Fdim = x_wo_aug.size()
        x_axis_size = int(math.sqrt(T_wo))

        if self.dynamic_subsampling:
            if self.energy_function == "random":
                self.energy = torch.rand(B, T_wo, device=x_wo_aug.device, dtype=x_wo_aug.dtype)
            elif self.energy_function == "power":
                self.energy = x_wo_aug.exp().sum(-1)
            elif self.energy_function == "local_linear":
                # Stage1: state estimation
                state = self.state_vectors(x_wo_aug.requires_grad_(False))
                # print(self.state_vectors.weight[0])
                # print(self.state_vectors.bias[:10])
                # print("+"*200)
                state = torch.nn.functional.softmax(state, dim=-1)  # (B, T, C)

                # Stage2: state prediction (vertical + horizontal)
                x_wo_aug_v = x_wo_aug.view(B, x_axis_size, x_axis_size, Fdim)
                x_wo_aug_v = x_wo_aug_v.transpose(1, 2).reshape(B, T_wo, Fdim)

                zero_pad = torch.zeros((B, 1, self.codebook_size), device=x.device, dtype=x.dtype)

                energy1 = self.energy_func1(x_wo_aug[:, 1:])         # right
                energy1[:, x_axis_size-1::x_axis_size] = zero_pad.repeat(1, energy1[:, x_axis_size-1::x_axis_size].size(1), 1)
                energy1 = torch.cat([energy1, zero_pad], dim=1)

                energy2 = self.energy_func2(x_wo_aug[:, 1:])         # right
                energy2[:, x_axis_size-1::x_axis_size] = zero_pad.repeat(1, energy2[:, x_axis_size-1::x_axis_size].size(1), 1)
                energy2 = torch.cat([zero_pad, energy2], dim=1)


                energy3 = self.energy_func3(x_wo_aug[:, 1:])         # right
                energy3[:, x_axis_size-1::x_axis_size] = zero_pad.repeat(1, energy3[:, x_axis_size-1::x_axis_size].size(1), 1)
                energy3 = torch.cat([energy3, zero_pad], dim=1)


                energy4 = self.energy_func4(x_wo_aug[:, 1:])         # right
                energy4[:, x_axis_size-1::x_axis_size] = zero_pad.repeat(1, energy4[:, x_axis_size-1::x_axis_size].size(1), 1)
                energy4 = torch.cat([zero_pad, energy4], dim=1)


                denom = torch.ones((energy4.size(0), energy4.size(1)),
                                   device=energy4.device, dtype=energy4.dtype) * 2
                denom[:, 0] = 1
                denom[:, -1] = 1

                state_pred_v = (energy1 + energy2) / denom.unsqueeze(-1)
                state_pred_v = state_pred_v.view(x_wo_aug.size(0), x_axis_size, x_axis_size, self.codebook_size)
                state_pred_v = state_pred_v.transpose(1, 2).reshape(x_wo_aug.size(0), x_wo_aug.size(1), self.codebook_size)


                state_pred = (energy3 + energy4) / denom.unsqueeze(-1)
                state_pred = (state_pred + state_pred_v) * 0.5
                state_pred = torch.nn.functional.softmax(state_pred, dim=-1)

                # Stage3: energy estimation
                self.energy = torch.sum(self.kl_loss(torch.log(state_pred + self.eps), state), dim=-1)

                # if print_option:
                #     print("state:", torch.argmax(state, dim=-1)[0])
                #     print("pred :", torch.argmax(state_pred, dim=-1)[0])

                # Stage4: loss calculation
                state_avg = torch.sum(state, dim=-1) / (T_wo + self.eps)
                self.entropy_maximization_loss = torch.sum(state_avg * torch.log(state_avg + self.eps), dim=-1)
                self.state_prediction_loss = torch.sum(-state * torch.log(state_pred + self.eps), dim = -1)
 
            else:
                raise NotImplementedError(f"Energy function {self.energy_function} is not implemented. ")


            if self.locality_constraint:


                odd_even = torch.randint(0, 2, (1,)).item()
                locality_mask = torch.arange(x_wo_aug.size(1), device=x.device)
                locality_mask = (locality_mask.view(x_axis_size, x_axis_size) + locality_mask.view(x_axis_size, x_axis_size).transpose(0,1)) % 2 == odd_even
                locality_mask = locality_mask.view(-1).unsqueeze(0)
                INF = torch.finfo(self.energy.dtype).max
                self.energy = self.energy.masked_fill(locality_mask, INF)


            # Stage6: prune the states
            vals, indices = self.energy.sort(dim=1, descending=True)
            indices = torch.cat([torch.zeros(indices.size(0), device=indices.device, dtype=indices.dtype).unsqueeze(-1), indices + 1], dim=1)
            if self.training:
                random_pruning_ratio = torch.rand(1).item() * self.max_pruning_ratio
                # print("random_pruning_ratio: ", random_pruning_ratio)
                # print("self.max_pruning_ratio: ", self.max_pruning_ratio)
                # print("x.size(1): ", x.size(1))
                
            else:
                random_pruning_ratio = self.max_pruning_ratio * 0.5

        state_to_remain = x.size(1) - int(random_pruning_ratio * x.size(1))
        if self.training:
            estimated_threshold = vals[:, state_to_remain - 2].mean()

        if self.aggregation == 'attention':
            x_before_pruning = x.clone()

        indices, _ = indices[:, :state_to_remain].sort(dim=1, descending=False)
        x = x.gather(1, indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        
        pos_embed = self.pos_embed.expand(B, -1, -1)
        if print_option and torch.isnan(pos_embed).any():
            print("pos_embed is nan before gather")
        # print("pos_embed: ", pos_embed.shape)
        pos_embed = pos_embed.gather(1, indices.unsqueeze(-1).expand(-1, -1, pos_embed.size(-1)))

        indices_wo_cls = indices[:, 1:] - 1

        if self.training:
            if self.energy_threshold == 0.0:
                with torch.no_grad():
                    self.energy_threshold.add_(estimated_threshold.detach())

            else:
                with torch.no_grad():
                    momentum = 0.99
                    self.energy_threshold.mul_(momentum).add_((1 - momentum) * estimated_threshold.detach())

        if self.aggregation == "cnn":
            pass

        elif self.aggregation == 'attention':
            # attention mask (2-dimensional)

            self.att_mask = self.att_mask.to(x.device)  # (B, T, T)
            att_mask = self.att_mask.gather(1, indices_wo_cls.unsqueeze(-1).expand(-1, -1, self.att_mask.size(-1)))  # (B, T', T)

            # cls token pad to attention mask
            att_mask = torch.cat([torch.zeros((att_mask.size(0), att_mask.size(1), 1), device = att_mask.device, dtype = att_mask.dtype), att_mask], dim=2)  # (B, T', T + 1)
            att_mask = torch.cat([torch.ones((att_mask.size(0), 1, att_mask.size(2)), device = att_mask.device, dtype = att_mask.dtype), att_mask], dim=1)  # (B, T' + 1, T + 1)
            if print_option and torch.isnan(att_mask).any():
                print('att_mask.size():', att_mask.size())

            # x size: B, T, F
            q = self.linear_q(x)  # x: (B, T' + 1, 768) q: (B, T' + 1, 64)
            k = self.linear_k(x_before_pruning)  # x_before_pruning: (B, T + 1, 768) k: (B, T + 1, 64)
            if print_option and torch.isnan(q).any():
                print("q is nan")
            if print_option and torch.isnan(k).any():
                print("k is nan")
            if print_option and torch.isinf(q).any():
                print("q is inf 1")
            if print_option and torch.isinf(k).any():
                print("k is inf 1")
            attn = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1)**0.5)  # (B, T' + 1, T + 1)
            if print_option and torch.isnan(attn).any():
                print("attn is nan 1")
            if print_option and torch.isinf(attn).any():
                print("attn is inf 1")
            self.l2_regularization_loss = (attn**2).mean()
            NEG_INF = torch.finfo(attn.dtype).min

            
            attn = attn.masked_fill((att_mask==0), NEG_INF)   # masked attention score (B, T' + 1, T + 1)
            before_softmax = attn.clone()  # TODO: REMOVE THIS LINE
            if print_option and torch.isnan(attn).any():
                print("attn is nan 2")
            if print_option and torch.isinf(attn).any():
                print("attn is inf 2")
            attn = attn.softmax(dim=-1)
            if print_option and torch.isnan(attn).any():
                print("attn is nan 3")

            if print_option and torch.isnan(x).any():
                print("x is nan before matmul with attn")
            x = torch.matmul(attn, x_before_pruning)                # (B, T' + 1, 768)
            if print_option and torch.isnan(x).any():
                print("x is nan after matmul with attn")
                print("random_pruning_ratio: ", random_pruning_ratio)
                print("self.max_pruning_ratio: ", self.max_pruning_ratio)
                print("x.size(1): ", x.size(1))
                print("state_to_remain : ", state_to_remain)
                import json
                with open("debug_nan.json", "w") as f:
                    json.dump({
                            #    "before_softmax" : before_softmax.detach().cpu().numpy().tolist(),
                            #    "attn": attn.detach().cpu().numpy().tolist(),
                            #    "x_before_pruning": x_before_pruning.detach().cpu().numpy().tolist(),
                            #    "x": x.detach().cpu().numpy().tolist(),
                               "q": q.detach().cpu().numpy().tolist(),
                               "k": k.detach().cpu().numpy().tolist(),
                               }, f)

                

        else:
            raise NotImplementedError(f"Energy function {self.energy_function} is not implemented.")

        # pos_embed = self.pos_embed[:, indices, :]
        # pos_embed = self.pos_embed.gather(1, indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        # pos_embed = self.pos_embed(indices)
        # pos_embed = pos_embed.gather(1, indices.unsqueeze(-1).expand(-1, -1, pos_embed.size(-1)))
        if print_option and torch.isnan(self.pos_embed).any():
            print("pos_embed is nan before posembed")

        if print_option and torch.isnan(pos_embed).any():
            print("pos_embed is nan after gather")

        if print_option and torch.isnan(x).any():
            print("x is nan before posembed")
        x = x + pos_embed
        if print_option and torch.isnan(x).any():
            print("x is nan after posembed")
        if print_option and torch.isinf(x).any():
            print("x is inf after posembed")

        x = self.pos_drop(x)

        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if print_option and torch.isnan(x).any():
                print("x is nan at ", idx, "in blk")
                print("pos_embed : ", pos_embed[0])
            if print_option and torch.isinf(x).any():
                print("x is inf at ", idx, "in blk")
                print("pos_embed : ", pos_embed[0])
            
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
            
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


    def get_energy_function_loss(self):
        if self.state_prediction_loss is not None and self.entropy_maximization_loss is not None:
            self.entropy_maximization_loss = self.entropy_maximization_loss.mean()
            self.state_prediction_loss = self.state_prediction_loss.sum(-1).mean()
            self.l2_regularization_loss = self.l2_regularization_loss.mean()
            # if torch.isnan(self.state_prediction_loss):
            #     print("state_prediction_loss is nan")
            #     sys.exit(1)
            # if torch.isnan(self.entropy_maximization_loss):
            #     print("entropy_maximization_loss is nan")
            #     sys.exit(1)

            return self.state_prediction_loss, self.entropy_maximization_loss, self.l2_regularization_loss

        else:
            return 0, 0

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model