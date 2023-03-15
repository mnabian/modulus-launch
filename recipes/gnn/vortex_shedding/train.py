# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch, dgl
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from dgl.dataloading import GraphDataLoader
from torch.cuda.amp import autocast, GradScaler
import time
import wandb as wb

try:
    import apex
except:
    pass

from modulus.models.gnn.meshgraphnet import MeshGraphNet
from modulus.datapipes.gnn.mgn_dataset import MGNDataset
from modulus.distributed.manager import DistributedManager


class MGNTrainer:
    def __init__(self, wb):
        # set distributed manager
        self.manager = DistributedManager()

        # set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")

        # instantiate dataset
        dataset = MGNDataset(
            name="vortex_shedding_train",
            data_dir=wb.config.data_dir,
            split="train",
            num_samples=wb.config.num_training_samples,
            num_steps=wb.config.num_training_time_steps,
        )

        # instantiate dataloader
        self.dataloader = GraphDataLoader(
            dataset,
            batch_size=wb.config.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

        # instantiate the model
        self.model = MeshGraphNet(
            8, 3, 1
        )  # this first two inputs (among 8 features) are x,y

        if wb.config.jit:
            self.model = torch.jit.script(self.model).to(self.device)
        else:
            self.model = self.model.to(self.device)
        if wb.config.watch_model:
            wb.watch(self.model)

        # enable train mode
        self.model.train()

        # instantiate loss, optimizer, and scheduler
        self.criterion = torch.nn.MSELoss()
        try:
            self.optimizer = apex.optimizers.FusedAdam(
                self.model.parameters(), lr=wb.config.lr
            )
            print("Using FusedAdam optimizer")
        except:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=wb.config.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: wb.config.lr_decay_rate**epoch
        )
        self.scaler = GradScaler()

        # load checkpoint
        self.load_checkpoint()

    def train(self, graph):
        graph = graph.to(self.device)
        self.optimizer.zero_grad()
        loss_data, loss_phys = self.forward(graph)
        loss = loss_data + loss_phys
        self.backward(loss)
        self.scheduler.step()
        return loss_data, loss_phys

    def load_checkpoint(self):
        # load checkpoint
        try:
            checkpoint = torch.load(wb.config.ckpt_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.epoch_init = checkpoint["epoch"]
            print(f"Successfully loaded checkpoint in {wb.config.ckpt_path}")
        except:
            self.epoch_init = -1

    def save_checkpoint(self, epoch):
        # save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            wb.config.ckpt_path,
        )
        print(f"saved model in {wb.config.ckpt_path}")

    def forward(self, graph):
        # forward pass
        with autocast(enabled=wb.config.amp):
            x = graph.ndata["mesh_pos"][:, 0:1].requires_grad_(True)
            y = graph.ndata["mesh_pos"][:, 1:2].requires_grad_(True)
            u, v, p = self.model(graph, x, y, graph.ndata["x"], graph.edata["x"])
            pred = torch.cat((u, v, p), dim=-1)
            loss_data = self.criterion(pred, graph.ndata["y"])

            # print(pred[:,0].size())
            # print(torch.squeeze(x).size())
            # exit()
            u__x = torch.autograd.grad(u, [x], grad_outputs=torch.ones_like(u), create_graph=True)[0]
            v__y = torch.autograd.grad(v, [y], grad_outputs=torch.ones_like(v), create_graph=True)[0]
            loss_phys = torch.mean((u__x + v__y)**2)
            return loss_data, loss_phys

    def backward(self, loss):
        # backward pass
        if wb.config.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()


if __name__ == "__main__":
    wb.init(project="VortexSheddingMGN", entity="modulus", mode="disabled")
    trainer = MGNTrainer(wb)
    start = time.time()
    print("Training started...")
    for epoch in range(trainer.epoch_init + 1, wb.config["epochs"]):
        for graph in trainer.dataloader:
            loss_data, loss_phys = trainer.train(graph)
        print(
            f"epoch: {epoch}, loss: {loss_data + loss_phys:10.3e}, loss_data: {loss_data:10.3e},, loss_phys: {loss_phys:10.3e}, time per epoch: {(time.time()-start):10.3e}"
        )
        wb.log({"loss": (loss_data + loss_phys).detach().cpu()})

        trainer.save_checkpoint(epoch)
        start = time.time()
    print("Training completed!")
