import sys

import os

sys.path.append('../../')
from fastreid.config import get_cfg
from fastreid.engine import default_argument_parser, default_setup, launch, DefaultTrainer
from fastreid.utils.checkpoint import Checkpointer
from fastreid.data import build_reid_train_loader

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    # dtld=build_reid_train_loader(cfg)
    # for i in dtl
    #     thisname=''
    #     for j in i['img_paths']:
    #         if thisname!='':
    #             if thisname!=j.split('/')[4]:
    #                 print("ERROR!!!!!!!")
    #         else :
    #             thisname=j.split('/')[4]
    #     print("    ",file=f)

    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = DefaultTrainer.build_model(cfg)
        Checkpointer(model, save_dir=cfg.OUTPUT_DIR).load(cfg.MODEL.WEIGHTS)  # load trained model

        res = DefaultTrainer.test(cfg, model)
        return res
    trainer = DefaultTrainer(cfg)

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )