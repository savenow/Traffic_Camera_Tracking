

# Predictions is the outputs of detectron2 and this manipulates data from it
boxes = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None
scores = predictions.scores if predictions.has("scores") else None
classes = predictions.pred_classes.numpy() if predictions.has("pred_classes") else None
masks = predictions.pred_masks.tensor.numpy()