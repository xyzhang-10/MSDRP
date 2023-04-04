import torch
import torch.nn as nn
import torch.nn.functional as F
class MSDRP(nn.Module):
    def __init__(self, drugs_dim, cells_dim,embed_dim, bathsize, dropout1, dropout2):
        super(MSDRP, self).__init__()
        self.drugs_dim = drugs_dim
        self.cells_dim = cells_dim
        self.batchsize = bathsize
        self.drug_dim = self.drugs_dim//12
        self.cell_dim = (self.cells_dim-580)//3
        self.total_layer_emb = 1536
        self.embed_dim = embed_dim
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.drugs_layer = nn.Linear(self.drugs_dim, self.embed_dim)
        self.drugs_layer_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.drugs_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)

        self.cells_layer = nn.Linear(self.cells_dim, self.embed_dim)
        self.cells_layer_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.cells_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)


        self.drug_layer1 = nn.Linear(self.drug_dim, self.embed_dim)
        self.drug_layer1_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.drug_layer2 = nn.Linear(self.drug_dim, self.embed_dim)
        self.drug_layer2_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.drug_layer3 = nn.Linear(self.drug_dim, self.embed_dim)
        self.drug_layer3_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.drug_layer4 = nn.Linear(self.drug_dim, self.embed_dim)
        self.drug_layer4_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.drug_layer5 = nn.Linear(self.drug_dim, self.embed_dim)
        self.drug_layer5_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.drug_layer6 = nn.Linear(self.drug_dim, self.embed_dim)
        self.drug_layer6_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.drug_layer7 = nn.Linear(self.drug_dim, self.embed_dim)
        self.drug_layer7_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.drug_layer8 = nn.Linear(self.drug_dim, self.embed_dim)
        self.drug_layer8_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.drug_layer9 = nn.Linear(self.drug_dim, self.embed_dim)
        self.drug_layer9_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.drug_layer10 = nn.Linear(self.drug_dim, self.embed_dim)
        self.drug_layer10_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.drug_layer11 = nn.Linear(self.drug_dim, self.embed_dim)
        self.drug_layer11_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.drug_layer12 = nn.Linear(self.drug_dim, self.embed_dim)
        self.drug_layer12_1 = nn.Linear(self.embed_dim, self.embed_dim)
        # self.drug_layer13 = nn.Linear(self.drug_dim, self.embed_dim)
        # self.drug_layer13_1 = nn.Linear(self.embed_dim, self.embed_dim)



        self.cell_layer1 = nn.Linear(self.cell_dim, self.embed_dim)
        self.cell_layer1_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.cell_layer2 = nn.Linear(self.cell_dim, self.embed_dim)
        self.cell_layer2_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.cell_layer3 = nn.Linear(self.cell_dim, self.embed_dim)
        self.cell_layer3_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.cell_layer4 = nn.Linear(580, self.embed_dim)
        self.cell_layer4_1 = nn.Linear(self.embed_dim, self.embed_dim)

        self.drug1_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.drug2_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.drug3_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.drug4_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.drug5_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.drug6_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.drug7_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.drug8_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.drug9_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.drug10_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.drug11_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.drug12_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        # self.drug13_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)




        self.cell1_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.cell2_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.cell3_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.cell4_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)


        # cnn setting
        self.channel_size =64
        self.number_map = 12 * 4

        ###外积残差块
        self.Outer_product_rb_1 = nn.Sequential(
            nn.Conv2d(self.number_map, self.channel_size * 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.channel_size * 4),
            nn.ReLU(),
            nn.Conv2d(self.channel_size * 4, self.channel_size, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.channel_size)
        )
        self.Outer_product_downsample = nn.Sequential(
            nn.Conv2d(self.number_map, self.channel_size, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.channel_size)
        )

        self.Outer_product_conv = nn.Sequential(
            nn.Conv2d(self.channel_size, self.channel_size, kernel_size=1, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=4, padding=1),
        )
        self.Outer_product_rb_2 = nn.Sequential(
            nn.Conv2d(self.channel_size, self.channel_size * 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.channel_size * 4),
            nn.ReLU(),
            nn.Conv2d(self.channel_size * 4, self.channel_size, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.channel_size)
        )
        self.Outer_product_maxpool = nn.MaxPool2d(kernel_size=3, stride=4, padding=1)
        self.Outer_product_maxpool1 = nn.MaxPool2d(kernel_size=3, stride=4, padding=1)
        self.re = nn.ReLU()

        self.Inner_Product_linear = nn.Sequential(
            nn.Linear(6144, 1024),
            nn.ELU(),
            nn.Dropout(self.dropout2),
            nn.Linear(1024, 1024),
            nn.ELU(),
            nn.Dropout(self.dropout2),
            nn.Linear(1024,512),
            nn.ELU(),
            nn.Dropout(self.dropout2),
            nn.Linear(512, 128)
        )

        self.total_layer = nn.Linear(self.total_layer_emb, self.channel_size * 4)
        self.total_bn = nn.BatchNorm1d((self.channel_size * 4 + 2 * self.embed_dim), momentum=0.5)
        self.con_layer =nn.Sequential(
            nn.Linear(self.channel_size * 4, 512),
            nn.ELU(),
            nn.Dropout(self.dropout2),
            nn.Linear(512, 1)
        )
        self.fused_network_layer = nn.Linear(170, self.embed_dim)
        self.fused_network_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.fused_network_layer_1 = nn.Linear(self.embed_dim, self.embed_dim)
    def forward(self, drug_features, cell_features, device,fused_network):

        fused_network= F.relu(self.fused_network_bn(self.fused_network_layer(fused_network.float().to(device))), inplace=True)
        fused_network= F.dropout(fused_network, training=self.training, p=self.dropout1)
        fused_network = self.fused_network_layer_1(fused_network)

        x_drugs = F.relu(self.drugs_bn(self.drugs_layer(drug_features.float().to(device))), inplace=True)
        x_drugs = F.dropout(x_drugs, training=self.training, p=self.dropout1)
        x_drugs = self.drugs_layer_1(x_drugs)

        x_cells = F.relu(self.cells_bn(self.cells_layer(cell_features.float().to(device))), inplace=True)
        x_cells = F.dropout(x_cells, training=self.training, p=self.dropout1)
        x_cells = self.cells_layer_1(x_cells)


        drug1, drug2, drug3, drug4, drug5, drug6, drug7, drug8, drug9, drug10, drug11, drug12= drug_features.chunk(12, 1)
        cell_ic_sim = cell_features[:, 2118:]
        cell_features = cell_features[:, :2118]
        cell1, cell2, cell3= cell_features.chunk(3, 1)

        x_drug1 = F.relu(self.drug1_bn(self.drug_layer1(drug1.float().to(device))), inplace=True)
        x_drug1 = F.dropout(x_drug1, training=self.training, p=self.dropout1)
        x_drug1 = self.drug_layer1_1(x_drug1)

        x_drug2 = F.relu(self.drug2_bn(self.drug_layer2(drug2.float().to(device))), inplace=True)
        x_drug2 = F.dropout(x_drug2, training=self.training, p=self.dropout1)
        x_drug2 = self.drug_layer2_1(x_drug2)

        x_drug3 = F.relu(self.drug3_bn(self.drug_layer3(drug3.float().to(device))), inplace=True)
        x_drug3 = F.dropout(x_drug3, training=self.training, p=self.dropout1)
        x_drug3 = self.drug_layer3_1(x_drug3)

        x_drug4 = F.relu(self.drug4_bn(self.drug_layer4(drug4.float().to(device))), inplace=True)
        x_drug4 = F.dropout(x_drug4, training=self.training, p=self.dropout1)
        x_drug4 = self.drug_layer4_1(x_drug4)

        x_drug5 = F.relu(self.drug5_bn(self.drug_layer5(drug5.float().to(device))), inplace=True)
        x_drug5 = F.dropout(x_drug5, training=self.training, p=self.dropout1)
        x_drug5 = self.drug_layer5_1(x_drug5)

        x_drug6 = F.relu(self.drug6_bn(self.drug_layer6(drug6.float().to(device))), inplace=True)
        x_drug6 = F.dropout(x_drug6, training=self.training, p=self.dropout1)
        x_drug6 = self.drug_layer6_1(x_drug6)

        x_drug7 = F.relu(self.drug7_bn(self.drug_layer7(drug7.float().to(device))), inplace=True)
        x_drug7 = F.dropout(x_drug7, training=self.training, p=self.dropout1)
        x_drug7 = self.drug_layer7_1(x_drug7)

        x_drug8 = F.relu(self.drug8_bn(self.drug_layer8(drug8.float().to(device))), inplace=True)
        x_drug8 = F.dropout(x_drug8, training=self.training, p=self.dropout1)
        x_drug8 = self.drug_layer8_1(x_drug8)

        x_drug9 = F.relu(self.drug9_bn(self.drug_layer9(drug9.float().to(device))), inplace=True)
        x_drug9 = F.dropout(x_drug9, training=self.training, p=self.dropout1)
        x_drug9 = self.drug_layer9_1(x_drug9)

        x_drug10 = F.relu(self.drug10_bn(self.drug_layer10(drug10.float().to(device))), inplace=True)
        x_drug10 = F.dropout(x_drug10, training=self.training, p=self.dropout1)
        x_drug10 = self.drug_layer10_1(x_drug10)

        x_drug11 = F.relu(self.drug11_bn(self.drug_layer11(drug11.float().to(device))), inplace=True)
        x_drug11 = F.dropout(x_drug11, training=self.training, p=self.dropout1)
        x_drug11 = self.drug_layer11_1(x_drug11)

        x_drug12 = F.relu(self.drug12_bn(self.drug_layer12(drug12.float().to(device))), inplace=True)
        x_drug12 = F.dropout(x_drug12, training=self.training, p=self.dropout1)
        x_drug12 = self.drug_layer12_1(x_drug12)





        drugs = [x_drug1, x_drug2, x_drug3, x_drug4, x_drug5, x_drug6, x_drug7, x_drug8, x_drug9, x_drug10, x_drug11, x_drug12]
        x_cell1 = F.relu(self.cell1_bn(self.cell_layer1(cell1.float().to(device))), inplace=True)
        x_cell1 = F.dropout(x_cell1, training=self.training, p=self.dropout1)
        x_cell1 = self.cell_layer1_1(x_cell1)

        x_cell2 = F.relu(self.cell2_bn(self.cell_layer2(cell2.float().to(device))), inplace=True)
        x_cell2 = F.dropout(x_cell2, training=self.training, p=self.dropout1)
        x_cell2 = self.cell_layer2_1(x_cell2)

        x_cell3 = F.relu(self.cell3_bn(self.cell_layer3(cell3.float().to(device))), inplace=True)
        x_cell3 = F.dropout(x_cell3, training=self.training, p=self.dropout1)
        x_cell3 = self.cell_layer3_1(x_cell3)

        x_cell4 = F.relu(self.cell4_bn(self.cell_layer4(cell_ic_sim.float().to(device))), inplace=True)
        x_cell4 = F.dropout(x_cell4, training=self.training, p=self.dropout1)
        x_cell4 = self.cell_layer4_1(x_cell4)

        cells = [x_cell1, x_cell2, x_cell3,x_cell4]


        #=============外积==========================================
        maps = []
        for i in range(len(drugs)):
            for j in range(len(cells)):
                maps.append(torch.bmm(drugs[i].unsqueeze(2), cells[j].unsqueeze(1)))
        Outer_product_map = maps[0].view((-1, 1, self.embed_dim, self.embed_dim))

        for i in range(1, len(maps)):
            interaction = maps[i].view((-1, 1, self.embed_dim, self.embed_dim))
            Outer_product_map = torch.cat([Outer_product_map, interaction], dim=1)
        #===============内积============================================
        total = []
        for i in range(len(drugs)):
            for j in range(len(cells)):
                total.append(drugs[i].unsqueeze(1)*cells[j].unsqueeze(1))

        Inner_Product_map =total[0]
        for i in range(1, len(maps)):
            Inner_Product_map= torch.cat([Inner_Product_map,total[i]], dim=1)
        #====================残差块=====================================================

        Inner_Product  = Inner_Product_map.view(Inner_Product_map.shape[0],-1)
        Inner_Product  =  self.Inner_Product_linear(Inner_Product )


        #########外积残差

        x = self.Outer_product_downsample(Outer_product_map)
        Outer_product_feature_map  =  self.Outer_product_rb_1(Outer_product_map)
        Outer_product_feature_map =Outer_product_feature_map + x
        Outer_product_feature_map= self.re(Outer_product_feature_map)

        Outer_product_feature_map = self.Outer_product_conv(Outer_product_feature_map)



        x = Outer_product_feature_map
        Outer_product_feature_map  = self.Outer_product_rb_2(Outer_product_feature_map)
        Outer_product_feature_map= Outer_product_feature_map+x
        Outer_product_feature_map = self.re(Outer_product_feature_map)

        Outer_product_feature_map = self.Outer_product_maxpool(Outer_product_feature_map)

        Outer_product= Outer_product_feature_map.view((x_drugs.shape[0], -1))


        total = torch.cat((x_drugs, Outer_product, Inner_Product,fused_network,x_cells), dim=1)

        total = F.relu(self.total_layer(total), inplace=True)
        total = F.dropout(total, training=self.training, p=self.dropout2)

        regression = self.con_layer(total)
        return  regression.squeeze()


