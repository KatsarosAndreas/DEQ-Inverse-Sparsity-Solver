import torch
import numpy as np
from solvers import new_equilibrium_utils as eq_utils
from torch import autograd
from utils import cg_utils
import gc
from tqdm import tqdm

def train_denoiser(denoising_net, train_dataloader, test_dataloader,
                 measurement_process, optimizer,
                 save_location, loss_function, n_epochs,
                 use_dataparallel=False, device='cpu', scheduler=None,
                 print_every_n_steps=10, save_every_n_epochs=5, start_epoch=0):

    for epoch in range(start_epoch, n_epochs):

        if epoch % save_every_n_epochs == 0:
            if use_dataparallel:
                torch.save({'solver_state_dict': denoising_net.module.state_dict(),
                            'epoch': epoch,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()
                            }, save_location)
            else:
                torch.save({'solver_state_dict': denoising_net.state_dict(),
                            'epoch': epoch,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()
                            }, save_location)

        # Create progress bar for current epoch
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), 
                   desc=f"Epoch {epoch+1}/{n_epochs}")
        
        for ii, sample_batch in pbar:
            optimizer.zero_grad()

            # Handle tuple format from MultiSliceFastMRIDataloader: (input_img, target_img)
            if isinstance(sample_batch, (tuple, list)):
                input_img, target_img = sample_batch
                sample_batch = target_img  # Use clean target for denoiser training
            
            sample_batch = sample_batch.to(device=device)
            y = measurement_process(sample_batch)
            reconstruction = y + denoising_net(y)
            loss = loss_function(reconstruction, sample_batch)
            loss.backward()
            optimizer.step()
            
            # Update progress bar with loss info
            current_loss = loss.cpu().detach().numpy()
            pbar.set_postfix({
                'Loss': f'{current_loss:.6f}',
                'Batch_Size': sample_batch.shape[0],
                'Step': ii
            })

            if ii % print_every_n_steps == 0:
                logging_string = "Epoch: " + str(epoch) + " Step: " + str(ii) + \
                                 " Loss: " + str(current_loss)
                print(logging_string, flush=True)

        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             print(type(obj), obj.size())
        #     except:
        #         pass
        # for name, val in denoising_net.named_parameters():
        #     print(name)
        # exit()
        if scheduler is not None:
            scheduler.step(epoch)
        if use_dataparallel:
            torch.save({'solver_state_dict': denoising_net.module.state_dict(),
                        'epoch': epoch,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()
                        }, save_location)
        else:
            torch.save({'solver_state_dict': denoising_net.state_dict(),
                        'epoch': epoch,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()
                        }, save_location)
