import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model
from tools import AverageMeter, knn_monitor, Logger, file_exist_check
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from linear_eval import main as linear_eval
from datetime import datetime
from contrastive_learning_dataset import ContrastiveLearningDataset
import time

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
# gpus = [0, 1]
# torch.cuda.set_device('cuda:{}'.format(gpus[0]))

def main(device, args):

    train_dataset = ContrastiveLearningDataset(txt_file="/home/polixir/yichao/Graduation_Project/dataset/mini-imagenet/train2.txt", \
                                               type="train", name='simsiam', img_size=84, train=True)

    # memory_dataset = ContrastiveLearningDataset(txt_file="/home/polixir/yichao/Graduation_Project/dataset/mini-imagenet/val1.txt", \
    #                                            type="train", name='simsiam', img_size=84, train=False, train_classifier=False)

    # test_dataset =  ContrastiveLearningDataset(txt_file="/home/polixir/yichao/Graduation_Project/dataset/mini-imagenet/test1.txt", \
    #                                            type="train", name='simsiam', img_size=84, train=False, train_classifier=False)                     

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )

    # memory_loader = torch.utils.data.DataLoader(
    #     dataset=memory_dataset,
    #     shuffle=False,
    #     batch_size=args.train.batch_size,
    #     **args.dataloader_kwargs
    # )

    # test_loader = torch.utils.data.DataLoader(
    #     dataset=test_dataset,
    #     shuffle=False,
    #     batch_size=args.train.batch_size,
    #     **args.dataloader_kwargs
    # )

    # train_loader = torch.utils.data.DataLoader(
        # dataset=get_dataset(
    #         transform=get_aug(train=True, **args.aug_kwargs), 
    #         train=True,
    #         **args.dataset_kwargs),
    #     shuffle=True,
    #     batch_size=args.train.batch_size,
    #     **args.dataloader_kwargs
    # )
    # memory_loader = torch.utils.data.DataLoader(
    #     dataset=get_dataset(
    #         transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), 
    #         train=True,
    #         **args.dataset_kwargs),
    #     shuffle=False,
    #     batch_size=args.train.batch_size,
    #     **args.dataloader_kwargs
    # )
    # test_loader = torch.utils.data.DataLoader(
    #     dataset=get_dataset( 
    #         transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), 
    #         train=False,
    #         **args.dataset_kwargs),
    #     shuffle=False,
    #     batch_size=args.train.batch_size,
    #     **args.dataloader_kwargs
    # )

    # define model
    model = get_model(args.model).to(device)
    model = torch.nn.DataParallel(model)
    # model = get_model(args.model)
    # model = nn.DataParallel(model, device_ids = gpus)
    # model = model.to(device)
    # model = 
    # model = torch.nn.DataParallel(model)

    # define optimizer
    optimizer = get_optimizer(
        args.train.optimizer.name, model, 
        lr=args.train.base_lr*args.train.batch_size/256, 
        momentum=args.train.optimizer.momentum,
        weight_decay=args.train.optimizer.weight_decay)

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.train.warmup_epochs, args.train.warmup_lr*args.train.batch_size/256, 
        args.train.num_epochs, args.train.base_lr*args.train.batch_size/256, args.train.final_lr*args.train.batch_size/256, 
        len(train_loader),
        constant_predictor_lr=True # see the end of section 4.2 predictor
    )

    logger = Logger(tensorboard=args.logger.tensorboard, matplotlib=args.logger.matplotlib, log_dir=args.log_dir)
    accuracy = 0 

    path1 = "/home/polixir/yichao/Graduation_Project/code/SimSiam/output/intrain_model_res12.pt"
    path2 = "/home/polixir/yichao/Graduation_Project/code/SimSiam/output/final_model_res12.pt"

    now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
    data_path = os.path.join("/home/polixir/yichao/Graduation_Project/code/SimSiam/output/train_data/", now + ".txt")

    data_file = open(data_path, "a+")
    # Start training
    global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training')
    min_loss = 10000
    for epoch in global_progress:
        model.train()
        
        # local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=args.hide_progress)
        local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=False)
        for idx, ((images1, images2), labels) in enumerate(local_progress):
        # for (images1, images2), labels in local_progress:

            model.zero_grad()
            data_dict = model.forward(images1.to(device, non_blocking=True), images2.to(device, non_blocking=True))
            # print("data_dict:", data_dict)
            loss = data_dict['loss'].mean() # ddp
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            data_dict.update({'lr':lr_scheduler.get_lr()})
            
            local_progress.set_postfix(data_dict)
            data_dict['loss'] = data_dict['loss'].mean()
            logger.update_scalers(data_dict)
        
        if(loss < min_loss):
            min_loss = loss
            best_path = f"/home/polixir/yichao/Graduation_Project/code/SimSiam/output/best_model.pt"
            torch.save(model.module.backbone.state_dict(), best_path) 

        if(epoch % 10 == 0):
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, path1)
        
        if((epoch+1) % 100 == 0):
            local_path = f"/home/polixir/yichao/Graduation_Project/code/SimSiam/output/model_epoch_{epoch+1}.pt"
            torch.save(model.module.backbone.state_dict(), local_path) 

        # if args.train.knn_monitor and epoch % args.train.knn_interval == 0: 
        #     accuracy = knn_monitor(model.backbone, memory_loader, test_loader, device, k=min(args.train.knn_k, len(memory_loader.dataset)), hide_progress=args.hide_progress) 
        
        epoch_dict = {"epoch":epoch, "accuracy":accuracy}
        data_file.write(str(epoch+1) + " " + str(loss.item()) + " " +\
                 str(accuracy) + "\n")
        print(f"\nEpoch: {epoch}\tLoss: {loss}")
        global_progress.set_postfix(epoch_dict)
        logger.update_scalers(epoch_dict)
    
    # Save checkpoint
    # model_path = os.path.join(args.ckpt_dir, f"{args.name}_{datetime.now().strftime('%m%d%H%M%S')}.pth") # datetime.now().strftime('%Y%m%d_%H%M%S')
    
    torch.save(model, path2) 
    # print(f"Model saved to {model_path}")
    # with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
    #     f.write(f'{model_path}')

    # if args.eval is not False:
    #     args.eval_from = model_path
    #     linear_eval(args)


if __name__ == "__main__":
    args = get_args()

    main(device=args.device, args=args)

    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')



    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')














