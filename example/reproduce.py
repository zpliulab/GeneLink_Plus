from Arguments import parser

if __name__ == '__main__':
    args = parser.parse_args()
    from pre_PBMC import pre_PBMCs
    from pre_PBMC_ex_create import hTFTarget_PCC
    from pre_tf_exPBMC import pre_data
    from Train_Test_Split import spilt
    from G_res4_v2_L2 import train
    from plote4 import con_net
    from out_st import st

    pre_PBMCs()
    hTFTarget_PCC(args)
    pre_data()
    spilt(args)
    train(args)
    con_net(args)
    st()