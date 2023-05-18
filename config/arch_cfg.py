import pathmagic

assert pathmagic


arch_config_comb1 = {
    "stage0": [
        [(7, 1), (1, 7)],
        [(1, 1)],
        [(5, 1), (1, 5)],
        [(1, 1)],
        # [(3, 1), (1, 3)],
        [(3, 3)],
    ],
    "stage1_residual": [
        [(7, 1), (1, 7)],
        [(3, 3)],
    ],
    "stage1_non_residual": [
        # [(3, 1), (1, 3)],
        [(3, 3)],
        [(5, 1), (1, 5)],
        [(1, 1)],
        [(7, 1), (1, 7)],
        [(1, 1)],
    ],
    "stage2_residual": [
        [(5, 1), (1, 5)],
        [(3, 3)],
    ],
    "stage2_non_residual": [
        # [(3, 3)],
        # [(3, 1), (1, 3)],
        [(3, 3)],
        [(5, 1), (1, 5)],
    ],
    "stage3_residual": [
        [(3, 3)],
        # [(3, 1), (1, 3)],
    ],
    "stage3_non_residual": [[(1, 1)]],
    "decoder_head_stage1": -1,
    # "decoder_head_stage2": -1,
    "decoder_head_stage2": -1,
    "decoder_head_stage3": -1,
}

arch_config_comb2 = {
    "stage0": [[(7, 1), (7, 1)]],
    "stage1_residual": [[(7, 7)]],
    "stage1_non_residual": [[(3, 3)], [(5, 1), (1, 5)], [(7, 1), (1, 7)]],
    "stage2_residual": [[(5, 1), (1, 5)]],
    "stage2_non_residual": [[(3, 1), (1, 3)], [(5, 5)]],
    "stage3_residual": [[(3, 1), (1, 3)]],
    "stage3_non_residual": [[(1, 1)]],
    "decoder_head_stage1": 0,
    "decoder_head_stage2": 1,
    "decoder_head_stage3": 1,
}

arch_config_comb3 = {
    "stage0": [[(7, 7)]],
    "stage1_residual": [[(7, 7)]],
    "stage1_non_residual": [[(3, 1), (1, 3)], [(5, 5)], [(7, 7)]],
    "stage2_residual": [[(5, 5)]],
    "stage2_non_residual": [[(3, 1), (1, 3)], [(5, 5)]],
    "stage3_residual": [[(3, 3)]],
    "stage3_non_residual": [[(1, 1)]],
    "decoder_head_stage1": 0,
    "decoder_head_stage2": -1,
    "decoder_head_stage3": -1,
}

arch_config_comb4 = {
    "stage0": [[(7, 1), (1, 7)]],
    "stage1_residual": [[(7, 1), (1, 7)]],
    "stage1_non_residual": [[(3, 3)], [(5, 1), (1, 5)], [(7, 1), (1, 7)]],
    "stage2_residual": [[(5, 1), (1, 5)]],
    "stage2_non_residual": [[(3, 3)], [(5, 1), (5, 1)]],
    "stage3_residual": [[(3, 3)]],
    "stage3_non_residual": [[(1, 1)]],
    "decoder_head_stage1": 0,
    "decoder_head_stage2": None,
    "decoder_head_stage3": -1,
}

arch_config_comb5 = {
    "stage0": [[(7, 1), (1, 7)]],
    "stage1_residual": [[(7, 7)]],
    "stage1_non_residual": [[(3, 1), (1, 3)],
                            [(5, 1), (1, 5)], [(7, 1), (1, 7)]],
    "stage2_residual": [[(5, 5)]],
    "stage2_non_residual": [[(3, 1), (1, 3)], [(5, 1), (5, 1)]],
    "stage3_residual": [[(3, 3)]],
    "stage3_non_residual": [[(1, 1)]],
    "decoder_head_stage1": 0,
    "decoder_head_stage2": -1,
    "decoder_head_stage3": -1,
}


arch_config_comb6 = {
    "stage0": [[(7, 7)]],
    "stage1_residual": [[(7, 7)]],
    "stage1_non_residual": [[(3, 3)], [(5, 5)], [(7, 7)]],
    "stage2_residual": [[(5, 5)]],
    "stage2_non_residual": [[(3, 1), (1, 3)], [(5, 1), (5, 1)]],
    "stage3_residual": [[(3, 3)]],
    "stage3_non_residual": [[(1, 1)]],
    "decoder_head_stage1": 0,
    "decoder_head_stage2": 0,
    "decoder_head_stage3": -1,
}


JETSEG_COMB = [
    arch_config_comb1, arch_config_comb2, arch_config_comb3,
    arch_config_comb4, arch_config_comb5, arch_config_comb6,
]
