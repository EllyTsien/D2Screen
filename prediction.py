import paddle.nn.functional as F
import paddle as pdl
import pandas as pd

class ModelTester:
    def __init__(self, model_class, data_loader_func, model_version):
        self.model_class = model_class
        self.data_loader_func = data_loader_func
        self.model_version = model_version
        self.model = self.load_model()
        self.test_data_loader = self.data_loader_func(mode='test', batch_size=1024)
        
    def load_model(self):
        model = self.model_class()
        model.set_state_dict(pdl.load("weight/" + self.model_version + ".pkl"))
        model.eval()
        return model

    def run_test(self):
        all_result = []
        for atom_bond_graph, bond_angle_graph, label_true_batch in self.test_data_loader:
            label_predict_batch = self.model(atom_bond_graph, bond_angle_graph)
            label_predict_batch = F.softmax(label_predict_batch)
            result = label_predict_batch[:, 1].cpu().numpy().reshape(-1).tolist()
            all_result.extend(result)

        df = pd.read_csv('data/data221048/test_nolabel.csv')
        df['pred'] = all_result
        df.to_csv('result.csv', index=False)

# Example usage:
# Assuming ADMET is your model class and get_data_loader is your data loader function
# model_tester = ModelTester(model_class=ADMET, data_loader_func=get_data_loader, model_version='1')
# model_tester.run_test()
