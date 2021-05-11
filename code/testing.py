from sklearn.metrics import f1_score


# TP = predicted as OOD and true label is OOD
# TN = predicted as IN and true label is IN
# FP = predicted as OOD and true label is IN
# FN = predicted as IN and true label is OOD

# FAR = Number of accepted OOD sentences / Number of OOD sentences
# FAR = FN / (TP + FN)

# FRR = Number of rejected ID sentences / Number of ID sentences
# FRR = FP / (FP + TN)


class Testing:
    """Used to test the results of classification."""

    def __init__(self, model, X_test, y_test, model_name: str, oos_label, bin_model=None, bin_oos_label=None):
        self.model = model
        self.X_test = X_test  # tf.Tensor
        self.y_test = y_test  # tf.Tensor
        self.oos_label = oos_label  # number
        self.model_name = model_name
        self.bin_model = bin_model
        self.bin_oos_label = bin_oos_label

    def test_train(self):
        accuracy_correct, accuracy_out_of = 0, 0

        pred_labels = self.model.predict(self.X_test)

        for pred_label, true_label in zip(pred_labels, self.y_test):
            if pred_label == true_label:
                accuracy_correct += 1

            accuracy_out_of += 1

        accuracy = accuracy_correct / accuracy_out_of * 100
        f1 = f1_score(self.y_test, pred_labels, average='macro') * 100

        return {'accuracy': round(accuracy, 1), 'f1': round(f1, 1)}
