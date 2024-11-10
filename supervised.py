"""
监督学习流程。划分数据集，创建网络和优化器，训练过程中每一个epoch结束会在验证集上测试。

提示：可以考虑按照下列方式修改代码：
1. 改变数据集划分、batch大小、学习率等超参数；
2. 从上一次训练的checkpoint加载模型接续训练；
3. 对训练进展进行监控和分析（如打印日志到tensorboard等）。
"""

from dataset import MahjongGBDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import CNNModel
import torch.nn.functional as F
import torch
import os

if __name__ == '__main__':
    logdir = 'model/'
    log_dir = 'model/log/run'
    if not os.path.exists(logdir + 'checkpoint'):
        os.mkdir(logdir + 'checkpoint')
    writer = SummaryWriter(log_dir)

    # Load dataset
    splitRatio = 0.9
    batchSize = 1024
    trainDataset = MahjongGBDataset(0, splitRatio, True)
    validateDataset = MahjongGBDataset(splitRatio, 1, False)
    loader = DataLoader(dataset = trainDataset, batch_size = batchSize, shuffle = True)
    vloader = DataLoader(dataset = validateDataset, batch_size = batchSize, shuffle = False)
    
    # Load model
    model = CNNModel().to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr = 5e-4)

    checkpoints = [f for f in os.listdir(logdir + 'checkpoint/') if f.endswith('.pkl')]
    checkpoints.sort(key=lambda x: int(x.split('.')[0]))  # Sort by epoch number
    if checkpoints:
        last_checkpoint = checkpoints[-1]
        epoch_to_resume = int(last_checkpoint.split('.')[0])  # Extract epoch number
        checkpoint_path = os.path.join(logdir + 'checkpoint/', last_checkpoint)
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        epoch_to_resume = 0
        print("Starting training from scratch.")
    
    # Train and validate
    for e in range(epoch_to_resume, 17):
        print(f'Resuming/Starting Epoch {e}')
        torch.save(model.state_dict(), logdir + 'checkpoint/%d.pkl' % e)

        # TensorBoard logs
        epoch_loss = 0.0
        writer.add_scalar('Loss/train', 0.0, global_step=e)

        for i, d in enumerate(loader):
            input_dict = {'is_training': True, 'obs': {'observation': d[0].cuda(), 'action_mask': d[1].cuda()}}
            logits = model(input_dict)
            loss = F.cross_entropy(logits, d[2].long().cuda())
            epoch_loss += loss.item()
            if i % 128 == 0:
                print('Iteration %d/%d'%(i, len(trainDataset) // batchSize + 1), 'policy_loss', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # TensorBoard logs
            writer.add_scalar('Loss/train', (epoch_loss / (i + 1)), global_step=e * len(loader) + i)

        print('Run validation:')
        correct = 0
        for i, d in enumerate(vloader):
            input_dict = {'is_training': False, 'obs': {'observation': d[0].cuda(), 'action_mask': d[1].cuda()}}
            with torch.no_grad():
                logits = model(input_dict)
                pred = logits.argmax(dim = 1)
                correct += torch.eq(pred, d[2].cuda()).sum().item()
        acc = correct / len(validateDataset)
        print('Epoch', e + 1, 'Validate acc:', acc)

        # TensorBoard logs
        writer.add_scalar('Accuracy/validation', acc, global_step=e)

    writer.close()       