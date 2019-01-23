from dreamai.model import *
from dreamai.fc import *
from dreamai.cv_model import *
from dreamai.recommender import *
from mylib.utils import *

def load_chkpoint(chkpoint_file):
        
    restored_data = torch.load(chkpoint_file)

    params = restored_data['params']
    print('load_chkpoint: best accuracy = {:.3f}'.format(params['best_accuracy']))  
    
    if params['model_type'].lower() == 'classifier':
        net = FC( num_inputs=params['num_inputs'],
                  num_outputs=params['num_outputs'],
                  layers=params['layers'],
                  criterion_name = params['criterion_name'],
                  optimizer_name = params['optimizer_name'],
                  model_name = params['model_name'],
                  lr = params['lr'],
                  dropout_p = params['dropout_p'],
                  best_accuracy = params['best_accuracy'],
                  best_validation_loss = params['best_validation_loss'],
                  best_model_file = params['best_model_file'],
                  chkpoint_file = params['chkpoint_file'],
                  class_names =  params['class_names'],
                  device=params['device']
          )

    elif params['model_type'].lower() == 'recommender':
        RecommenderNet(
                 num_users=params['n_users'],
                 num_items=params['n_items'],
                 layers=params['layers'],
                 embedding_dim=params['embedding_dim'],
                 criterion_name = params['criterion_name'],
                 optimizer_name=params['optimizer_name'],
                 dropout_p=params['dropout_p'],
                 embedding_dropout_p=params['embedding_dropout_p'],
                 hidden_non_linearity=params['hidden_non_linearity'],
                 output_non_linearity=parmas['output_non_linearity'],
                 model_type =params['model_type'],
                 model_name=params['model_name'],
                 best_validation_loss = params['best_validation_loss'],
                 best_model_file = params['best_model_file'],
                 chkpoint_file = params['chkpoint_file'],
                 device=params['device'])



    elif params['model_type'].lower() == 'cv_transfer':
        net = TransferNetworkImg(criterion_name = params['criterion_name'],
                                 optimizer_name = params['optimizer_name'],
                                 model_name = params['model_name'],
                                 lr = params['lr'],
                                 device=params['device'],
                                 dropout_p = params['dropout_p'],
                                 best_accuracy = params['best_accuracy'],
                                 best_accuracy_file = params['best_accuracy_file'],
                                 chkpoint_file = params['chkpoint_file'],
                                 head = params['head']
                               )
    
        


    net.load_state_dict(torch.load(params['best_accuracy_file']))

    net.to(params['device'])
    
    return net
    
