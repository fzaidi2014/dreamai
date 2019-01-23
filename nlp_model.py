import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from dreamai.utils import *
from dreamai.data import *
from dreamai.model import *
import time
import copy


def load_CharRNN(fname):
    with open(fname, 'rb') as f:
        checkpoint = torch.load(f)

    char_rnn = CharLevelModel(checkpoint['vocab_size'],n_hidden=checkpoint['n_hidden'],
                              embedding_dim=checkpoint['embedding_dim'],
                              n_layers=checkpoint['n_layers'],
                              criterion=nn.CrossEntropyLoss(),
                              drop_prob=0.5,
                              lr=0.001)
    char_rnn.load_state_dict(checkpoint['state_dict'])
    return char_rnn



class CharLevelModel(Network):
    
    def __init__(self, vocab_size, embedding_dim=None,model_type='LSTM',
                 n_hidden=256, n_layers=2,
                 drop_prob=0.25,criterion=nn.NLLLoss(),optimizer='Adam',lr=0.001,
                 fc_layers=[],
                 chkpoint_file = 'chkpoint_cnn.pth',
                 best_valid_file = 'best_valid.pth',
                 best_accuracy_file ='best_accuracy.pth'):
        super().__init__()
        #print('criterion = {}'.format(criterion))
        self.set_model_params(criterion,optimizer,chkpoint_file, best_valid_file,best_accuracy_file,lr,drop_prob)

        self.char2int = None
        self.int2char = None
        self.chars = None

        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.embedding = None
        self.embedding_dim = None
        self.vocab_size = vocab_size
        
        if embedding_dim is not None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.embedding.to(self.device)
            input_dim = embedding_dim
            self.embedding_dim = embedding_dim
        else:
            input_dim = vocab_size
        

        if model_type.lower() == 'lstm':
            self.model = nn.LSTM(input_dim, n_hidden, n_layers,
                                dropout=drop_prob, batch_first=True)
        
        self.dropout = nn.Dropout(drop_prob)
        self.fc = FC(n_hidden,vocab_size,criterion=criterion,
                     layers=fc_layers,lr=lr,dropout=drop_prob,model_type='predictor')

        self.model.to(self.device)
        self.fc.to(self.device)
        
    def save(self,fname):
        checkpoint = {
            'n_hidden': self.n_hidden,
            'n_layers': self.n_layers,
            'embedding_dim':self.embedding_dim,
            'state_dict': self.state_dict(),
            'chars': self.chars,
            'vocab_size':self.vocab_size
        }

        with open(fname, 'wb') as f:
            torch.save(checkpoint, f)


               
    def forward(self, x, hidden):

        if self.embedding is not None:
            x = self.embedding(x)
        else:
            x = one_hot_encode(x, self.vocab_size).to(self.device)      

        rnn_output, hidden = self.model(x,hidden)
        out = self.dropout(rnn_output)
        out = out.contiguous().view(-1, self.n_hidden)
        out = self.fc(out)
        return out, hidden
    
    
    def _init_hidden(self,batch_size):
        # Initializes hidden state
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        hidden = (weight.new(self.n_layers,batch_size, self.n_hidden).zero_().to(self.device),
                  weight.new(self.n_layers,batch_size, self.n_hidden).zero_().to(self.device))
        
        return hidden

    def _train(self,dataset,clip=5,print_every=10):
        num_batches = 0
        train_data = dataset.train
        batch_size = dataset.batch_size
        train_data_size = dataset.get_train_size()
        sequence_length = dataset.sequence_length
        running_loss = 0
        h = self._init_hidden(batch_size)
        
        train_batches = dataset.get_batches(train_data['batches'])
        for inputs, labels in train_batches:
            num_batches += 1
            #t1 = time.time()
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            h = tuple([each.data for each in h])
            self.optimizer.zero_grad()
            h = tuple([each.data for each in h])
            output, h = self.forward(inputs, h)
            loss = self.criterion(output, labels.view(batch_size*sequence_length))
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), clip)
            self.optimizer.step()
            #print('training this batch took {:.3f} seconds'.format(time.time() - t1))
            running_loss += loss.item()
        return running_loss,num_batches   
            
                
                
       
    def _validate(self,dataset):
        losses = []
        valid_data = dataset.valid
        batch_size = dataset.batch_size
        sequence_length = dataset.sequence_length
        val_h = self._init_hidden(dataset.batch_size)
        valid_batches = dataset.get_batches(valid_data['batches'])
        for inputs, labels in valid_batches:
            val_h = tuple([each.data for each in val_h])
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            output, val_h = self.forward(inputs, val_h)
            loss = self.criterion(output, labels.view(batch_size*sequence_length))
            losses.append(loss.item())
            del inputs
            del labels
            
        return (np.mean(losses))

    def fit(self,dataset,epochs=2,print_every=10,es=False,change_dropout=False,clip=5,
            chkpoint_every=10,validate_every=1,save_best=False,early_stopping={'train_loss':5,'valid_loss':5}):
        
        import math
        best_validation_loss = math.inf
        best_accuracy = 0
        
        self.char2int = dataset.char2int
        self.int2char = dataset.int2char
        self.chars = dataset.chars

        prev_train_loss = math.inf
        prev_valid_loss = math.inf
        prev_accuracy = 0
        es_train_loss_count = 0
        es_valid_loss_count = 0
        es_accuracy_count = 0
        es_overfit_count = 0
        self.best_accuracy_model = self.model.state_dict()
        self.best_validation_model = self.model.state_dict()

        batch_size = dataset.batch_size
        
        set_optimizer(self,self.parameters(),optimizer=self.optimizer,lr=self.lr)
        
        for epoch in range(epochs):
            t0 = time.time()
            self.model.to(self.device)
            print('moving to cuda took {:.3f} seconds'.format(time.time() - t0))
            self.train()
            t0 = time.time()
            running_loss,num_batches = self._train(dataset,clip=clip,
                                               print_every=print_every)
            #if chkpoint_every > 0 and self.chkpoint_file is not None and epoch % chkpoint_every == 0:
            #    self.save()
            if num_batches % print_every == 0:
                print(f"{time.asctime()}"
                        f"Time Elapsed = {time.time()-t0:.3f}.."
                        f"Epoch {epoch+1}/{epochs}.. "
                        f"Batch {batches+1}/{dataset.get_train_size()}.. "
                        f"Average Training loss: {running_loss/(batches):.3f}.. "
                        f"Batch Training loss: {loss:.3f}.. "
                        )
            epoch_train_loss = running_loss/dataset.get_train_size()
            if epoch % validate_every == 0:
                
                self.eval()
                with torch.no_grad():
                    t2 = time.time()
                    epoch_validation_loss = self._validate(dataset)
                    time_elapsed = time.time() - t2
                    print(f"{time.asctime()}--Validation took {time.time() - t2:.3f} seconds.."
                          f"Epoch {epoch+1}/{epochs}.. "
                          f"Epoch Training loss: {epoch_train_loss:.3f}.. "
                          f"Epoch validation loss: {epoch_validation_loss:.3f}.. ")
                    
                if change_dropout:        
                    p = self._get_dropout()    
                    if (epoch_validation_loss >= epoch_train_loss) and ((p + 0.05) <= 0.5):
                        print('Overfitting detected: changing dropout probability {:.3f} to {:.3f}'.format(p,p+0.05))
                        self._set_dropout(p=p+0.05)

                    elif (p - 0.05) > 0.:

                        print('Underfitting detected: changing dropout probability {:.3f} to {:.3f}'.format(p,p-0.05))
                        self._set_dropout(p=p-0.05)
                
                    
                self.train()
                if es:
                    prev_valid_loss,es_valid_loss_count,action = check_early_stopping(prev_valid_loss,epoch_validation_loss,es_valid_loss_count,early_stopping,'valid_loss')
                    if action == 'stop':
                        print('stopping early due to matching valid loss criteria {}'.format(early_stopping['valid_loss']))
                        break
                    
                if save_best:
                    #self.model.to('cpu')
                    print(best_validation_loss,epoch_validation_loss)
                    #if best_validation_loss is not None:
                    #    print('best validation loss = {:.3f}'.format(best_validation_loss))
                    #print('best validation accuracy = {:.3f}'.format(best_accuracy))
                    #best_validation_loss =  save_best_model(self,best_validation_loss,epoch_validation_loss)
            if es:
                prev_train_loss,es_train_loss_count,action = check_early_stopping(prev_train_loss,epoch_train_loss,es_train_loss_count,early_stopping,'train_loss')
                if action == 'stop':
                    print('stopping early due to matching training loss criteria {}'.format(early_stopping['train_loss']))
                    break
                                         
                
        if save_best:
            print('loading best accuracy model')
            self.model.load_state_dict(self.best_accuracy_model)

    def predict(self, char, h=None, top_k=None):
        
        if self.char2int is None or self.int2char is None or self.chars is None:
            print('the char model has not been trianed yet..please call fit first before predcting')
            return

        self.eval()
        with torch.no_grad():
            # tensor inputs
            x = torch.tensor([[self.char2int[char]]]).to(self.device)

            # detach hidden state from history
            h = tuple([each.data for each in h])

            out, h = self.forward(x, h)

        # get the character probabilities
        p = F.softmax(out, dim=1).data
        p = p.cpu() # move to cpu
        
        # get top characters
        if top_k is None:
            top_ch = np.arange(len(self.chars))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()
        
        # select the likely next character with some element of randomness
        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p/p.sum())
        
        # return the encoded value of the predicted char and the hidden state
        return self.int2char[char], h

    def sample(self,size, prime='The', top_k=None):
        
        # First off, run through the prime characters
        chars = [ch for ch in prime]
        h = self._init_hidden(1)
        for ch in prime:
            char, h = self.predict(ch, h, top_k=top_k)

        chars.append(char)
        
        # Now pass in the previous character and get a new one
        for ii in range(size):
            char, h = self.predict(chars[-1], h, top_k=top_k)
            chars.append(char)

        return ''.join(chars)         

class LanguageModel(nn.Module):
    def __init__(self,ntokens,ninp,nhid,nlayers,dropout=0.5,criterion=nn.CrossEntropyLoss(),lr=0.5,clip=2,
                 tie_weights=False,optimizer=None):
        super().__init__()
        self.device = 'cuda'
        self.drop = nn.Dropout()
        self.encoder = nn.Embedding(ntokens,ninp)
        self.rnn = nn.LSTM(ninp,nhid,nlayers,dropout=dropout)
        self.decoder = nn.Linear(nhid,ntokens)
        self.criterion = criterion
        self.lr = lr
        self.clip = clip
        self.ntokens = ntokens
        
        
        if tie_weights:
            self.decoder.weight = self.encoder.weight
        
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers
        
        self.rnn = self.rnn.to('cuda')
        self.encoder = self.encoder.to('cuda')
        self.decoder = self.decoder.to('cuda')
        
        if optimizer is not None:
            set_optimizer(self,self.parameters(),optimizer=optimizer,lr=self.lr)
        else:
            self.optimizer = None
        
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange,initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange,initrange)
        
    def forward(self,input,hidden): 
        
        emb = self.drop(self.encoder(input))
        output,hidden = self.rnn(emb,hidden)
        output = self.drop(output)
        s = output.size()
        decoded = self.decoder(output.view(s[0]*s[1],s[2]))
        return decoded.view(s[0],s[1],decoded.size(1)),hidden
    
    def init_hidden(self,bsz):
        weight = next(self.parameters()).data
        return (weight.new(self.nlayers,bsz,self.nhid).zero_().to('cuda'),
                weight.new(self.nlayers,bsz,self.nhid).zero_().to('cuda'))
    
    def repackage_hidden(self,h):
        return tuple([each.data for each in h])
    
    def clip_gradients(self,clip):
        torch.nn.utils.clip_grad_norm_(self.parameters(),clip)
        
    def validate_(self,lmdata,log_every=100):
        self.eval()
        total_loss = 0   
        hidden = self.init_hidden(lmdata.batch_size)
        batch_count = 0
        for i,batch in enumerate(lmdata.valid_iter):        
            data, targets = batch.text,batch.target.view(-1)
            output, hidden = self.forward(data, hidden)
            hidden = self.repackage_hidden(hidden)
            loss = self.criterion(output.view(-1, self.ntokens), targets).data
            total_loss += loss.item()
            if i%log_every == 0 and i > 0:
                print('validating: batch = {}, average loss = {:.3f}'.format(i+1,total_loss/i))
            batch_count += 1
        self.train()
        return total_loss/batch_count 
    
    def train_(self,lmdata,log_every=200):
    
        # Turn on training mode which enables dropout.
        self.train()
        total_loss = 0
        start_time = time.time()
        hidden = self.init_hidden(lmdata.batch_size)
        for  i,batch in enumerate(lmdata.train_iter):
            data, targets = batch.text,batch.target.view(-1)
            data,targets = data.to('cuda'),targets.to('cuda')
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = self.repackage_hidden(hidden)
            self.zero_grad()
            output, hidden = self.forward(data, hidden)
            loss = self.criterion(output.view(-1, self.ntokens), targets)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            self.clip_gradients(self.clip)
            
            if self.optimizer is None:
                for p in self.parameters():
                    p.data.add_(-self.lr, p.grad.data)
            else:
                self.optimizer.step()

            total_loss += loss.item()

            if i % log_every == 0 and i > 0:
                cur_loss = total_loss / log_every
                elapsed = time.time() - start_time
                print('| {:5d}/{:5d} batches | lr {:02.2f} | time spent in this interval {:5.2f} secs | loss {:5.2f} | ppl {:8.2f}'.format(i,
                     len(lmdata.train_iter),self.lr,elapsed,cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()
                
                
    def fit_(self,lmdata,epochs,train_log_every=200,validate_log_every=20,adapt_lr=True):
        min_val_loss = None
            
        for epoch in range(1, epochs+1):
            epoch_start_time = time.time()
            self.train_(lmdata,log_every=train_log_every)
            val_loss = self.validate_(lmdata,log_every=validate_log_every)
            print('-' * 89)
            print('| end of epoch {:3d}/{:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch,epochs,
                             (time.time() - epoch_start_time),
                              val_loss, math.exp(val_loss)))
            if adapt_lr and self.optimizer is None:
                if not min_val_loss or val_loss < min_val_loss:
                    min_val_loss = val_loss
                else:
                    self.lr /= 4.0
                    print('updated learning rate: new learning rate = {}'.format(self.lr))
    
