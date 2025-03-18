# Datasets and Model Training

## Datasets Download

- **VGGFace2**: Download from [Kaggle](https://www.kaggle.com/datasets/dimarodionov/vggface2)  

- **CASIA**: Download from [Drive](https://drive.google.com/file/d/1A9tijVZYYt5bbIXwXK7Ud-LOnGfvXF50/view?usp=sharing)  

## Models Download
- Download from [Drive](https://drive.google.com/drive/folders/1xtYJXiWTcX6cpZiRU8wTOubYWae-iJeu?usp=drive_link)

---

## Model Training

### Self-Trained Classification Models for VGGFace2

**Training Command**:
```bash
python ./self-train_VGGFace2/train_efficientnet_b0.py
```

**Defense Training Command**:
```bash
python ./self-train_VGGFace2/train_model_defense.py
```

### Parameter Description
| Parameter | Type | Description |
|------|------|-------|
| `--defense_method` | str | Optional：`BiDO`/`MID`/`LS`/`TL` |

#### Hyperparameters of the defense
| Defenses | Parameter | type |
|---------|------|------|
| **BiDO** | `--coef_hidden_input` | float |
|          | `--coef_hidden_output` | float |
| **MID**  | `--beta` | float |
| **LS**   | `--coef_label_smoothing` | float |
| **TL**   | `--layer_name` | str |
---

## Acknowledge
The defenses are implemented based on the following repositories. We extend our gratitude to the authors for open-sourcing their code.

[BiDO](https://github.com/AlanPeng0897/Defend_MI), [MID](https://github.com/Jiachen-T-Wang/mi-defense), [LS](https://github.com/LukasStruppek/Plug-and-Play-Attacks), [TL](https://github.com/hosytuyen/TL-DMI), [MIA-ToolBox](https://github.com/ffhibnese/Model-Inversion-Attack-ToolBox)

