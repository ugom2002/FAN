# models.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

####################################
# Définition de la couche FANLayer
####################################
class FANLayer(nn.Module):
    """
    Couche FAN (Fourier Analysis Network)
    """
    def __init__(self, input_dim, output_dim, p_ratio=0.25, activation='gelu', use_p_bias=True):
        super(FANLayer, self).__init__()
        # p_ratio doit être dans (0, 0.5)
        assert 0 < p_ratio < 0.5, "p_ratio must be between 0 and 0.5"
        self.p_ratio = p_ratio
        p_output_dim = int(output_dim * self.p_ratio)
        # La branche p génère cos et sin donc on compte deux fois
        g_output_dim = output_dim - 2 * p_output_dim
        self.input_linear_p = nn.Linear(input_dim, p_output_dim, bias=use_p_bias)
        self.input_linear_g = nn.Linear(input_dim, g_output_dim)
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation if activation else lambda x: x

    def forward(self, src):
        # src: (batch, d_model)
        g = self.activation(self.input_linear_g(src))
        p = self.input_linear_p(src)
        # Concaténation de cosinus, sinus et de la branche g
        output = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)
        return output

####################################
# Blocs Transformer
####################################

# Bloc Transformer avec FFN classique
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, src):
        # src: (seq_len, batch, d_model)
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# Bloc Transformer avec une couche FAN en remplacement du FFN
class TransformerFANBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, fan_p_ratio=0.25, dropout=0.1):
        super(TransformerFANBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # La FANLayer prend en entrée des vecteurs de dimension d_model et produit un vecteur de dimension d_ff.
        self.fan_layer = FANLayer(d_model, d_ff, p_ratio=fan_p_ratio, activation='gelu')
        self.linear_out = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # src: (seq_len, batch, d_model)
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # Appliquer la FANLayer token par token :
        # on transpose pour obtenir (batch, seq_len, d_model), puis on reshape en (batch*seq_len, d_model)
        bsz = src.size(1)
        seq_len = src.size(0)
        src_reshaped = src.transpose(0, 1).reshape(-1, src.size(2))
        fan_out = self.fan_layer(src_reshaped)  # (batch*seq_len, d_ff)
        fan_out = fan_out.reshape(bsz, seq_len, -1).transpose(0, 1)  # (seq_len, batch, d_ff)
        fan_out = self.linear_out(fan_out)
        src = src + self.dropout2(fan_out)
        src = self.norm2(src)
        return src

####################################
# Encodage positionnel (Positional Encoding)
####################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

####################################
# Modèle Transformer pour la prévision de séries temporelles
####################################
class TransformerForecast(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, d_ff, pred_len, dropout=0.1, use_fan=False, fan_p_ratio=0.25):
        """
        input_size: nombre de features en entrée (par exemple 7 pour un dataset multivarié)
        pred_len: longueur de la fenêtre de prédiction
        Le modèle prédit pour chaque feature, donc la dimension de sortie sera input_size * pred_len.
        """
        super(TransformerForecast, self).__init__()
        self.input_size = input_size
        self.pred_len = pred_len
        self.input_linear = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.use_fan = use_fan
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if use_fan:
                self.layers.append(TransformerFANBlock(d_model, nhead, d_ff, fan_p_ratio, dropout))
            else:
                self.layers.append(TransformerBlock(d_model, nhead, d_ff, dropout))
        # La couche de décodage prédit l'ensemble des valeurs futures
        self.decoder = nn.Linear(d_model, input_size * pred_len)
    
    def forward(self, src):
        # src: (batch, seq_len, input_size)
        src = self.input_linear(src)  # (batch, seq_len, d_model)
        src = src.transpose(0, 1)  # (seq_len, batch, d_model)
        src = self.pos_encoder(src)
        for layer in self.layers:
            src = layer(src)
        # Utilisation de la représentation du dernier pas de temps
        out = src[-1]  # (batch, d_model)
        out = self.decoder(out)  # (batch, input_size * pred_len)
        # Reshape pour obtenir (batch, pred_len, input_size)
        out = out.view(-1, self.pred_len, self.input_size)
        return out
