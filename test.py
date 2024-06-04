import os
import numpy as np
from sklearn.metrics import confusion_matrix
import cv2

def calculate_metrics(y_true, y_pred):
    # Flatten the arrays to compare pixel-wise
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Calculate confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true_flat, y_pred_flat, labels=[0, 255]).ravel()

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0  # Sensitivity is the same as recall
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    f1 = (2 * tp) / ((2 * tp) + fp + fn) if ((2 * tp) + fp + fn) != 0 else 0
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) != 0 else 0

    return accuracy, precision, sensitivity, specificity, f1, jaccard

def evaluate_single_mask(pred_path, gt_path):
    # Read the images
    pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    # Ensure the masks have the same shape
    if pred_mask.shape == gt_mask.shape:
        # Map the ground truth skin pixels to background
        gt_mask[gt_mask == 127] = 0

        # Calculate the metrics
        acc, prec, sens, spec, f1, jacc = calculate_metrics(gt_mask, pred_mask)

        metrics = {
            'accuracy': acc,
            'precision': prec,
            'sensitivity': sens,
            'specificity': spec,
            'f1_score': f1,
            'jaccard_index': jacc
        }

        return metrics
    else:
        print(f"Warning: Shape mismatch between {pred_path} and {gt_path}")
        return None

def evaluate_masks(pred_dir, gt_dir):
    metrics_list = []
    for gt_filename in os.listdir(gt_dir):
        if gt_filename.endswith('.png'):
            gt_path = os.path.join(gt_dir, gt_filename)
            pred_filename_png = gt_filename.replace('truth', 'pred')
            pred_filename_jpg = pred_filename_png.replace('.png', '.jpg')

            pred_path_png = os.path.join(pred_dir, pred_filename_png)
            pred_path_jpg = os.path.join(pred_dir, pred_filename_jpg)

            if os.path.exists(pred_path_png):
                metrics = evaluate_single_mask(pred_path_png, gt_path)
                if metrics:
                    metrics_list.append(metrics)
            elif os.path.exists(pred_path_jpg):
                metrics = evaluate_single_mask(pred_path_jpg, gt_path)
                if metrics:
                    metrics_list.append(metrics)
            else:
                print(f"Warning: Predicted mask not found for ground truth {gt_filename}")

    return metrics_list

def calculate_average_metrics(metrics_list):
    if not metrics_list:
        return {}

    avg_metrics = {
        'accuracy': np.mean([metrics['accuracy'] for metrics in metrics_list]),
        'precision': np.mean([metrics['precision'] for metrics in metrics_list]),
        'sensitivity': np.mean([metrics['sensitivity'] for metrics in metrics_list]),
        'specificity': np.mean([metrics['specificity'] for metrics in metrics_list]),
        'f1_score': np.mean([metrics['f1_score'] for metrics in metrics_list]),
        'jaccard_index': np.mean([metrics['jaccard_index'] for metrics in metrics_list]),
    }

    return avg_metrics

if __name__ == "__main__":
    pred_dir = "tests/pred"
    gt_dir = "tests/truth"

    metrics_list = evaluate_masks(pred_dir, gt_dir)

    if metrics_list:
        avg_metrics = calculate_average_metrics(metrics_list)
        print("Average Evaluation Metrics:")
        for metric, value in avg_metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
    else:
        print("No valid metrics calculated.")
