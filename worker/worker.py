import torch
import torch.nn.functional as F

def Model_Train(model, optimizer, scheduler, data_iter, device, loss, l_c, l_a, acc, configs):
    model.train()
    loss.reset()
    l_c.reset()
    l_a.reset()
    acc.reset()
    for x_t, x_f, _ in data_iter:
        optimizer.zero_grad()
        x_t, x_f = x_t.to(device), x_f.to(device)
        logits_t, logits_f, z = model(x_t, x_f)
        batch_size= x_t.shape[0]
        labels_con = torch.arange(batch_size, device=device, dtype=torch.long)
        labels_ali = torch.cat([torch.ones(batch_size, dtype=torch.long), torch.zeros(2*batch_size, dtype=torch.long)], dim=0).to(device)
        l_con= (
            F.cross_entropy(logits_t, labels_con) + 
            F.cross_entropy(logits_f, labels_con) 
        ) * (2 * configs.T)
        l_ali = F.cross_entropy(z, labels_ali)
        l = l_con + l_ali
        l.backward()
        optimizer.step()
        with torch.no_grad():
            loss.update((l, batch_size))
            l_c.update((l_con, batch_size))
            l_a.update((l_ali, batch_size))
            acc.update((z, labels_ali))
    scheduler.step()
    return loss.compute(), l_c.compute(), l_a.compute(), acc.compute()

def Model_Finetune(model, optimizer, scheduler, data_iter, device, loss, acc):
    model.train()
    loss.reset()
    acc.reset()
    for x_t, x_f, y in data_iter:
        optimizer.zero_grad()
        x_t, x_f, y = x_t.to(device), x_f.to(device), y.to(device)
        y_hat = model(x_t, x_f)
        batch_size= x_t.shape[0]
        l = F.cross_entropy(y_hat, y)
        l.backward()
        optimizer.step()
        with torch.no_grad():
            loss.update((l, batch_size))
            acc.update((y_hat, y))
    scheduler.step()
    total_loss = loss.compute()
    total_acc = acc.compute()
    return total_loss, total_acc

def Model_Test(model, data_iter, device, acc):
    model.eval()
    acc.reset()
    with torch.no_grad():
        for x_t, x_f, y in data_iter:
            x_t, x_f, y= x_t.to(device), x_f.to(device), y.to(device)
            y_hat = model(x_t, x_f)
            acc.update((y_hat, y))
        total_acc = acc.compute()
    return total_acc
