# Perception Guide

## Segmentation

### 3D U-Net

```python
from rhenium.perception.segmentation import UNet3D

model = UNet3D(
    in_channels=1,
    out_channels=4,  # background + 3 classes
    features=[32, 64, 128, 256],
    use_attention=True,
)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
for batch in dataloader:
    logits = model(batch["image"])
    loss = dice_loss(logits, batch["mask"])
    loss.backward()
    optimizer.step()
```

### UNETR (Transformer)

```python
from rhenium.perception.segmentation import UNETR

model = UNETR(
    img_size=(96, 96, 96),
    patch_size=16,
    embed_dim=768,
    num_heads=12,
)
```

---

## Detection

### CenterNet3D

Anchor-free detection for lesions/nodules:

```python
from rhenium.perception.detection import CenterNet3D, Detection

model = CenterNet3D(num_classes=1)

outputs = model(volume_tensor)
detections: list[Detection] = model.decode_detections(
    outputs,
    threshold=0.3,
    input_shape=volume.shape,
)

for det in detections:
    print(f"Center: {det.center}, Size: {det.size}, Score: {det.score}")
```

---

## Classification

### Volume-Level Classification

```python
from rhenium.perception.classification import ResNet3D

model = ResNet3D(
    in_channels=1,
    num_classes=5,
    layers=[2, 2, 2, 2],
)

probs = model.predict_proba(volume_tensor)
predicted_class = probs.argmax(dim=1)
```

### Ordinal Classification (Grading)

For ordered scales like PI-RADS, BI-RADS:

```python
from rhenium.perception.classification import OrdinalClassifier

model = OrdinalClassifier(
    backbone=resnet_backbone,
    num_grades=5,
)

grade_probs = model(volume_tensor)
# Output: probability for each grade 1-5
```

---

## Metrics

### Segmentation

| Metric | Formula | Range |
|--------|---------|-------|
| Dice | \(\frac{2\|P \cap G\|}{\|P\| + \|G\|}\) | [0, 1] |
| IoU | \(\frac{\|P \cap G\|}{\|P \cup G\|}\) | [0, 1] |
| HD95 | 95th percentile Hausdorff | [0, ∞) mm |

```python
from rhenium.evaluation import dice_score, iou_score, hausdorff_distance_95

dice = dice_score(pred_mask, gt_mask)
iou = iou_score(pred_mask, gt_mask)
hd95 = hausdorff_distance_95(pred_mask, gt_mask)
```

### Classification

```python
from rhenium.evaluation import auroc, expected_calibration_error

auc = auroc(pred_probs, labels)
ece = expected_calibration_error(pred_probs, labels)
```
