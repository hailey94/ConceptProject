import os
import torch
from termcolor import colored
from configs import parser
from utils.engine_mine import train, test
from model.partdetection.model_main import MMDiscovery
from loaders.get_loader import loader_generation
from utils.tools import fix_parameter, print_param

os.makedirs('saved_model/', exist_ok=True)


def set_seed(seed):
    print(f"Set random seed as {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def main():
    model = MMDiscovery(args)
    device = torch.device(args.device)

    # CUDNN
    torch.backends.cudnn.benchmark = True

    if not args.pre_train:
        if args.dataset !='imagenet':

            checkpoint = torch.load(os.path.join(args.output_dir,
                                                 f"{args.dataset}_{args.base_model}_cls{args.num_classes}_cpt_no_slot.pt"),
                                    map_location=device)
            model.load_state_dict(checkpoint, strict=False)
            fix_parameter(model, ["layer1", "layer2", "back_bone.conv1", "back_bone.bn1"], mode="fix")
            print(colored('trainable parameter name: ', "blue"))
            print_param(model)
        print("load pre-trained model finished, start training")
    else:
        print("start training the backbone")

    #
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params=params)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, 0.1)

    model.to(device)

    train_loader1, train_loader2, val_loader = loader_generation(args)
    print("data prepared")
    acc_max = 0
    cnt = 0

    for i in range(args.epoch):
        print(colored('Epoch %d/%d' % (i + 1, args.epoch), 'yellow'))
        print(colored('-' * 15, 'yellow'))


        train(args, model, device, train_loader1, optimizer, i)
        scheduler.step()
        acc = test(args, model, val_loader, device)

        # print(acc)
        if acc > acc_max:
            acc_max = acc
            cnt = 0
            print(f"get better result, save current model., acc: {100*acc}")
            # op_dir = os.path.join(args.output_dir)
            torch.save(model.state_dict(), os.path.join(args.output_dir,
                                                        f"{args.dataset}_{args.base_model}_cls{args.num_classes}_" + f"cpt{args.num_cpt}_" +
                                                        f"{'use_slot_' + args.cpt_activation}_{args.name}.pt"))
        else:
            cnt += 1


if __name__ == '__main__':
    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir + '/', exist_ok=True)
    main()
