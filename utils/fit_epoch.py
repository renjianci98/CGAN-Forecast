import os

import torch
from tqdm import tqdm


def fit_one_epoch(G, D, G_optimizer, D_optimizer, G_steps,D_steps, loss_fn, batch_size, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch,  save_period, weight_save_dir, G_loss_history, D_loss_history):
    D_train_loss = 0
    G_train_loss = 0
    D_val_loss = 0
    G_val_loss = 0
    print('Start Train')
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            for i in range(D_steps):
                D_optimizer.zero_grad()
                history_data, forecast_gt = batch
                real_score = D(history_data, forecast_gt)
                D_real_loss = loss_fn(
                    real_score, torch.ones_like(real_score, device='cuda'))
                D_real_loss.backward()
                noise = torch.randn((batch_size, 128), device='cuda')
                forecast_result = G(history_data, noise)
                fake_score = D(history_data, forecast_result.detach())
                D_fake_loss = loss_fn(fake_score,
                                    torch.zeros_like(fake_score, device='cuda'))
                D_fake_loss.backward()
                # 判别器损失
                D_loss = D_real_loss + D_fake_loss
                # 判别器优化
                D_optimizer.step()
            for i in range(G_steps):
                G_optimizer.zero_grad()
                noise = torch.randn((batch_size, 128), device='cuda')
                forecast_result = G(history_data, noise)
                fake_score = D(history_data, forecast_result)
                # 生成器损失
                G_loss = loss_fn(fake_score,
                                 torch.ones_like(fake_score, device='cuda'))
                G_loss.backward()
                # 生成器优化
                G_optimizer.step()
            with torch.no_grad():
                D_train_loss += D_loss.item()
                G_train_loss += G_loss.item()
            D_steps=max(1,int((D_train_loss-G_train_loss)*5))
            G_steps=max(1,int((G_train_loss-D_train_loss)*5))

            pbar.set_postfix(**{'d_loss': D_train_loss / (iteration + 1),
                                'g_loss': G_train_loss / (iteration + 1),
                                })
            pbar.update(1)

    print('Finish Train')
    
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            history_data, forecast_gt = batch
            with torch.no_grad():
                D_optimizer.zero_grad()
                G_optimizer.zero_grad()
                real_score = D(history_data, forecast_gt)
                D_real_loss = loss_fn(
                    real_score, torch.ones_like(real_score, device='cuda'))
                noise = torch.randn((batch_size, 128), device='cuda')
                forecast_result = G(history_data, noise)
                fake_score = D(history_data, forecast_result)
                D_fake_loss = loss_fn(
                    fake_score, torch.zeros_like(fake_score, device='cuda'))
                D_loss = D_real_loss+D_fake_loss
                G_loss = loss_fn(fake_score,
                                    torch.ones_like(fake_score, device='cuda'))
                D_val_loss += D_loss.item()
                G_val_loss += G_loss.item()
                pbar.set_postfix(
                    **{'d_val_loss': D_val_loss / (iteration + 1),
                        'g_val_loss': G_val_loss / (iteration + 1)
                        })
                pbar.update(1)

    print('Finish Validation')

    D_loss_history.append_loss(
        epoch + 1, D_train_loss / epoch_step, D_val_loss / epoch_step_val)
    G_loss_history.append_loss(
        epoch + 1, G_train_loss / epoch_step, G_val_loss / epoch_step_val)
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('D loss: %.3f || G loss:: %.3f || D val Loss: %.3f || G val Loss: %.3f ' %
            (D_train_loss/epoch_step, G_train_loss/epoch_step, D_val_loss / epoch_step_val, G_val_loss / epoch_step_val))
    if epoch + 1 == Epoch:
        torch.save(G.module.state_dict(), os.path.join(weight_save_dir, 'final.pth'))
    elif (epoch + 1) % save_period == 0:
        torch.save(G.module.state_dict(), os.path.join(weight_save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' %
                   (epoch + 1, G_train_loss / epoch_step, G_val_loss / epoch_step_val)))
    
    return D_steps,G_steps
