import numpy as np
import torch
import math
from torch import nn
from collections import OrderedDict
# from model.sparsemax import Sparsemax


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self,
                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        patch_tokens = x[:, 1:, :]

        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj

        # return x
        # return patch_tokens
        return x, patch_tokens


class TextTransformer(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int):
        super().__init__()
        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.positional_embedding.dtype

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        stroke_tokens = x

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        # return x
        # return stroke_tokens
        return x, stroke_tokens


class SignWithSigmoidGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        result = (x > 0).float()
        sigmoid_result = torch.sigmoid(x)
        ctx.save_for_backward(sigmoid_result)
        return result

    @staticmethod
    def backward(ctx, grad_result):
        (sigmoid_result,) = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = grad_result * sigmoid_result * (1 - sigmoid_result)
        else:
            grad_input = None
        return grad_input


class Clip_Codebook(nn.Module):
    def __init__(self,
                 vision_input_resolution: int,
                 vision_patch_size: int,
                 vision_width: int,
                 vision_layers: int,
                 vision_heads: int,
                 vision_output_dim: int,
                 text_embed_dim: int,
                 text_context_length: int,
                 text_vocab_size: int,
                 text_width: int,
                 text_heads: int,
                 text_layers: int,
                 codebook_num: int,
                 codebook_dim: int,
                 temperature: int,
                 is_threshold: bool):
        super().__init__()

        self.visual = VisionTransformer(
            input_resolution=vision_input_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=vision_output_dim
        )
        self.encode_text = TextTransformer(
            embed_dim=text_embed_dim,
            context_length=text_context_length,
            vocab_size=text_vocab_size,
            transformer_width=text_width,
            transformer_heads=text_heads,
            transformer_layers=text_layers,
        )
        self.context_length = text_context_length
        self.codebook_num = codebook_num
        self.codebook_dim = codebook_dim

        # codebook
        self.codebook = nn.Parameter(torch.randn(self.codebook_num, self.codebook_dim))

        self.temperature = temperature

        # map patch/text tokens to codebook(query) spaces
        self.img_q_map = nn.Sequential(
            nn.LayerNorm(vision_width),
            nn.Linear(vision_width, codebook_dim),
            nn.GELU(),
            nn.LayerNorm(codebook_dim),
            nn.Linear(codebook_dim, codebook_dim)
        )
        self.txt_q_map = nn.Sequential(
            nn.LayerNorm(text_width),
            nn.Linear(text_width, codebook_dim),
            nn.GELU(),
            nn.LayerNorm(codebook_dim),
            nn.Linear(codebook_dim, codebook_dim)
        )

        self.att_activation = nn.Softmax(dim=-1)
        # self.att_activation = Sparsemax(dim=-1)

        self.softmax = nn.Softmax(dim=1)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.min_value = -1e+6
        self.threshold = 0.0
        self.is_threshold = is_threshold

    @property
    def dtype(self):
        try:
            return self.visual.conv1.weight.dtype
        except:
            try:
                return self.visual.head.weight.dtype
            except:
                try:
                    return self.visual.stem[0].weight.dtype
                except:
                    return self.encode_text.text_projection.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def val_encode_text(self, text):
        text_global_features, text_tokens = self.encode_text(text)        # [batch, stroke_token, text_width] [b, 30, 512]
        txt_q = self.txt_q_map(text_tokens)     # [batch, stroke_token, codebook_dim] [b, 30 ,256]

        k = self.codebook
        k = k.unsqueeze(0)      # [1, codebook_num, codebook_dim]
        k = k.transpose(2, 1)   # [1, codebook_dim, codebook_num]

        txt_inner_dot = torch.matmul(txt_q, k)  # [batch, stroke_token, code_num] [b, 30, 200]
        txt_inner_dot = txt_inner_dot / math.sqrt(self.codebook_dim)  # scale dot norm

        txt_mask = torch.zeros(len(text), self.context_length)
        txt_mask = txt_mask.cuda()
        txt_mask = txt_mask.masked_fill(text == 0, float("-inf"))
        assert txt_mask.shape == txt_q.shape[:2]
        mask = (txt_mask == 0) * 1  # 0 --> 1, inf --> 0
        txt_inner_dot = txt_inner_dot * mask.unsqueeze(-1)

        # temperature norm
        # txt_inner_dot = txt_inner_dot / self.temperature    # [batch, stroke_num, code_num]

        txt_inner_dot_ = txt_inner_dot.max(1)[0]    # [batch, codebook_num]
        if self.is_threshold:
            mask = SignWithSigmoidGrad.apply(txt_inner_dot_)
            txt_inner_dot_ = txt_inner_dot_.masked_fill((1 - mask).bool(), float('-inf'))
        txt_att_weight = self.att_activation(txt_inner_dot_)  # [batch, codebook_num]
        text_local_features = txt_att_weight @ self.codebook      # [batch, codebook_dim]

        return text_local_features, text_global_features

    def val_encode_data(self, image, text_local, text_global):
        text_local_features = text_local
        text_global_features = text_global

        image_global_features, image_tokens = self.encode_image(image)     # [batch, patch_token, vision_width] [b, 64, 256]
        img_q = self.img_q_map(image_tokens)    # [batch, patch_token, codebook_dim] [b, 64, 256]

        k = self.codebook
        k = k.unsqueeze(0)      # [1, codebook_num, codebook_dim]
        k = k.transpose(2, 1)   # [1, codebook_dim, codebook_num]

        img_inner_dot = torch.matmul(img_q, k)  # [batch, patch_token, code_num] [b, 64, 200]
        img_inner_dot = img_inner_dot / math.sqrt(self.codebook_dim)  # scale dot norm

        # temperature norm
        # img_inner_dot = img_inner_dot / self.temperature    # [batch, patch_num, code_num]

        img_inner_dot_ = img_inner_dot.max(1)[0]    # [batch, codebook_num]
        if self.is_threshold:
            mask = SignWithSigmoidGrad.apply(img_inner_dot_)
            img_inner_dot_ = img_inner_dot_.masked_fill((1 - mask).bool(), float('-inf'))
        img_att_weight = self.att_activation(img_inner_dot_)  # [batch, codebook_num]
        image_local_features = img_att_weight @ self.codebook     # [batch, codebook_dim]

        # Local
        image_local_features = image_local_features / image_local_features.norm(dim=1, keepdim=True)
        text_local_features = text_local_features / text_local_features.norm(dim=1, keepdim=True)
        logit_scale = self.logit_scale.exp()

        logits_local_per_image = logit_scale * image_local_features @ text_local_features.t()
        # logits_local_per_text = logit_scale * text_local_features @ image_local_features.t()

        # Global
        image_global_features = image_global_features / image_global_features.norm(dim=1, keepdim=True)
        text_global_features = text_global_features / text_global_features.norm(dim=1, keepdim=True)

        logits_global_per_image = logit_scale * image_global_features @ text_global_features.t()
        # logits_global_per_text = logit_scale * text_global_features @ image_global_features.t()

        return logits_local_per_image, logits_global_per_image

    def forward(self, image, text):
        image_global_features, image_tokens = self.encode_image(image)     # [batch, patch_token, vision_width] [b, 64, 256]
        text_global_features, text_tokens = self.encode_text(text)        # [batch, stroke_token, text_width] [b, 30, 512]
        img_q = self.img_q_map(image_tokens)    # [batch, patch_token, codebook_dim] [b, 64, 256]
        txt_q = self.txt_q_map(text_tokens)     # [batch, stroke_token, codebook_dim] [b, 30 ,256]

        k = self.codebook
        k = k.unsqueeze(0)      # [1, codebook_num, codebook_dim]
        k = k.transpose(2, 1)   # [1, codebook_dim, codebook_num]

        img_inner_dot = torch.matmul(img_q, k)  # [batch, patch_token, code_num] [b, 64, 200]
        txt_inner_dot = torch.matmul(txt_q, k)  # [batch, stroke_token, code_num] [b, 30, 200]
        img_inner_dot = img_inner_dot / math.sqrt(self.codebook_dim)  # scale dot norm
        txt_inner_dot = txt_inner_dot / math.sqrt(self.codebook_dim)  # scale dot norm

        txt_mask = torch.zeros(len(text), self.context_length)
        txt_mask = txt_mask.cuda()
        txt_mask = txt_mask.masked_fill(text == 0, float("-inf"))
        assert txt_mask.shape == txt_q.shape[:2]
        mask = (txt_mask == 0) * 1  # 0 --> 1, inf --> 0
        txt_inner_dot = txt_inner_dot * mask.unsqueeze(-1)

        # temperature norm
        # img_inner_dot = img_inner_dot / self.temperature    # [batch, patch_num, code_num]
        # txt_inner_dot = txt_inner_dot / self.temperature    # [batch, stroke_num, code_num]

        # 转为codebook维度
        img_inner_dot_ = img_inner_dot.max(1)[0]    # [batch, codebook_num]
        if self.is_threshold:
            mask = SignWithSigmoidGrad.apply(img_inner_dot_)
            img_inner_dot_ = img_inner_dot_.masked_fill((1 - mask).bool(), float('-inf'))
            # print(mask[0])
            # count = (1-mask[0]).sum().item()
            # print('count:', count)
        img_att_weight = self.att_activation(img_inner_dot_)  # [batch, codebook_num]
        # print(img_att_weight[0])
        # img_masked_count = (1 - (img_att_weight > 0).float()).sum().item()
        # print('img_avg_token:', img_att_weight.shape[1] - img_masked_count / len(img_att_weight))
        image_local_features = img_att_weight @ self.codebook     # [batch, codebook_dim]

        txt_inner_dot_ = txt_inner_dot.max(1)[0]    # [batch, codebook_num]
        # print(txt_inner_dot_)
        if self.is_threshold:
            mask = SignWithSigmoidGrad.apply(txt_inner_dot_)
            txt_inner_dot_ = txt_inner_dot_.masked_fill((1 - mask).bool(), float('-inf'))
            # count = mask.sum().item()
            # print('count:', count)
        txt_att_weight = self.att_activation(txt_inner_dot_)  # [batch, codebook_num]
        # txt_masked_count = (1 - (txt_att_weight > 0).float()).sum().item()
        # print('txt_avg_token:', txt_att_weight.shape[1] - txt_masked_count / len(txt_att_weight))
        # exit()
        text_local_features = txt_att_weight @ self.codebook      # [batch, codebook_dim]

        # print('img:\n', img_inner_dot.size(), '\n', img_inner_dot_[0])
        # print('img:\n', img_att_weight.size(), '\n', img_att_weight[0])
        # print('txt:\n', txt_inner_dot.size(), '\n', txt_inner_dot[0][0])

        # Local
        image_local_features = image_local_features / image_local_features.norm(dim=1, keepdim=True)
        text_local_features = text_local_features / text_local_features.norm(dim=1, keepdim=True)
        logit_scale = self.logit_scale.exp()

        logits_local_per_image = logit_scale * image_local_features @ text_local_features.t()
        # logits_local_per_text = logit_scale * text_local_features @ image_local_features.t()

        # Global
        image_global_features = image_global_features / image_global_features.norm(dim=1, keepdim=True)
        text_global_features = text_global_features / text_global_features.norm(dim=1, keepdim=True)

        logits_global_per_image = logit_scale * image_global_features @ text_global_features.t()
        # logits_global_per_text = logit_scale * text_global_features @ image_global_features.t()

        # return logits_local_per_image
        return logits_local_per_image, logits_global_per_image


    def visualization(self, image):
        image_global_features, image_tokens = self.encode_image(image)
        img_q = self.img_q_map(image_tokens)

        k = self.codebook
        k = k.unsqueeze(0)
        k = k.transpose(2, 1)

        img_inner_dot = torch.matmul(img_q, k)
        img_inner_dot = img_inner_dot / math.sqrt(self.codebook_dim)

        img_inner_dot_ = img_inner_dot.max(1)[0]
        img_att_weight = self.att_activation(img_inner_dot_)
        image_local_features = img_att_weight @ self.codebook

        image_local_features = image_local_features / image_local_features.norm(dim=1, keepdim=True)
        image_global_features = image_global_features / image_global_features.norm(dim=1, keepdim=True)

        return image_local_features, image_global_features