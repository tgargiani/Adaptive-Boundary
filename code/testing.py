from utils import get_f1_ood_id

from sklearn.metrics import f1_score, accuracy_score


class Testing:
    """Used to test the results of classification."""

    def __init__(self, model, X_test, y_test, model_name: str, oos_label):
        self.model = model
        self.X_test = X_test  # tf.Tensor
        self.y_test = y_test  # tf.Tensor
        self.model_name = model_name
        self.oos_label = oos_label  # number

    def test_train(self):
        pred_labels = self.model.predict(self.X_test)

        accuracy_all = accuracy_score(self.y_test, pred_labels) * 100
        f1_all = f1_score(self.y_test, pred_labels, average='macro') * 100
        f1_ood, f1_id = get_f1_ood_id(self.y_test, pred_labels, self.oos_label)

        return {'accuracy_all': round(accuracy_all, 1), 'f1_all': round(f1_all, 1),
                'f1_ood': round(f1_ood, 1), 'f1_id': round(f1_id, 1)}
