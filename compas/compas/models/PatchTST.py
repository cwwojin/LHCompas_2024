from tsai.models.PatchTST import PatchTST as Model
from yacs.config import CfgNode as CN


class PatchTST(Model):
    """
    Wrapper Class for PatchTST model.

    Parameters
    ---
    cfg : CfgNode
        Configuration object
    """

    def __init__(self, cfg: CN):
        super().__init__(
            c_in=cfg.C_IN,  # number of input channels
            c_out=cfg.C_OUT,  # used for compatibility
            seq_len=cfg.SEQ_LEN,  # input sequence length
            pred_dim=cfg.PRED_DIM,  # prediction sequence length
            n_layers=cfg.N_LAYERS,  # number of encoder layers
            n_heads=cfg.N_HEADS,  # number of heads
            d_model=cfg.D_MODEL,  # dimension of model
            d_ff=cfg.D_FF,  # dimension of fully connected network (fcn)
            dropout=cfg.DROPOUT,  # dropout applied to all linear layers in the encoder
            attn_dropout=cfg.ATTN_DROPOUT,  # dropout applied to the attention scores
            patch_len=cfg.PATCH_LEN,  # patch_len
            stride=cfg.STRIDE,  # stride
            padding_patch=cfg.PADDING_PATCH,  # flag to indicate if padded is added if necessary
            revin=cfg.REVIN,  # RevIN
            affine=cfg.AFFINE,  # RevIN affine
            individual=cfg.INDIVIDUAL,  # individual head
            subtract_last=cfg.SUBTRACT_LAST,  # subtract_last
            decomposition=cfg.DECOMPOSITION,  # apply decomposition
            kernel_size=cfg.KERNEL_SIZE,  # decomposition kernel size
            activation=cfg.ACTIVATION,  # activation function of intermediate layer, relu or gelu.
            norm=cfg.NORM,  # type of normalization layer used in the encoder
            pre_norm=cfg.PRE_NORM,  # flag to indicate if normalization is applied as the first step in the sublayers
            res_attention=cfg.RES_ATTENTION,  # flag to indicate if Residual MultiheadAttention should be used
            store_attn=cfg.STORE_ATTN,  # can be used to visualize attention weights
        )
        self.config = cfg
