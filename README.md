# Driving Perception Lab

Application Streamlit de démonstration et de comparaison des deux modèles du PFE :

`ImageNet ou Driving-Aware I-JEPA → ViT-S/16 → SFP P2-P6 → Faster R-CNN`

L'interface reprend les codes visuels d'une console de perception automobile :
vue caméra centrale, bounding boxes lisibles, registre des objets, télémétrie et
fiche scientifique du modèle. Il s'agit d'une analyse d'images, pas d'une
connexion directe au simulateur CARLA et pas d'un système de conduite réel.

## Résumé du modèle

| Élément | Valeur |
|---|---|
| Dataset | KITTI, split train 5 984 / validation 1 497 |
| Résolution | 1024 × 320 |
| Backbone | ViT-S/16, 384 dimensions |
| Pyramide | P2 à P6 |
| Détecteur | Faster R-CNN |
| Classes | Pedestrian, Cyclist, Car, Van |
| Checkpoints | ImageNet époque 25 · I-JEPA époque 35 |

Trois modes sont disponibles : **ImageNet baseline**, **Driving-Aware I-JEPA**
et **comparaison côte à côte** sur la même image.

## 1. Fabriquer le ZIP final dans Colab

Le ZIP source ne contient volontairement pas les deux checkpoints privés.

1. Importez `01_PREPARER_DEPLOIEMENT_COLAB.ipynb` dans Google Colab.
2. Exécutez la cellule 1 et sélectionnez le ZIP source du projet.
3. Montez Google Drive dans la cellule 2.
4. Vérifiez le chemin :

```text
/content/drive/MyDrive/PFE_JEPA/NB6_FINAL_COLAB/detection_E2_architecture/checkpoints/best_detector.pt
/content/drive/MyDrive/PFE_JEPA/NB2_imagenet_baseline/checkpoints/best_detector.pt
```

5. Exécutez les cellules restantes.

Le notebook :

- extrait uniquement `model_state_dict` ;
- stocke les tenseurs flottants en FP16 pour réduire la taille ;
- découpe le fichier en morceaux de 20 MiB ;
- crée un manifeste SHA-256 ;
- reconstruit strictement l'architecture et vérifie le checkpoint ;
- télécharge `DRIVING_PERCEPTION_LAB_FINAL.zip`.

Le notebook accepte le runtime actuel de Google Colab, notamment Python 3.13.
Il ne tente pas d'y installer les roues Linux/Python 3.12 réservées à Streamlit.

La taille totale d'un modèle peut dépasser 100 Mo : ce n'est pas un problème,
car seuls les morceaux de 20 MiB sont envoyés sur GitHub.

## 2. Envoyer le projet sur GitHub

Décompressez `DRIVING_PERCEPTION_LAB_FINAL.zip`. N'envoyez jamais le ZIP
lui-même dans le dépôt. Envoyez son contenu décompressé.

La racine GitHub doit contenir directement :

```text
app.py
model.py
inference.py
checkpoint_files.py
requirements.txt
style.css
.streamlit/config.toml
models/ijepa/checkpoint_manifest.json
models/ijepa/best_detector.pt.part001
models/ijepa/best_detector.pt.part002
models/imagenet/checkpoint_manifest.json
models/imagenet/best_detector.pt.part001
models/imagenet/best_detector.pt.part002
...
```

Tous les morceaux font moins de 25 Mo et peuvent être importés avec l'interface
web GitHub. Aucun Git LFS n'est nécessaire.

## 3. Déployer sur Streamlit Community Cloud

Dans <https://share.streamlit.io/> :

```text
Repository       : votre dépôt GitHub
Branch           : main
Main file path   : app.py
Python version   : 3.12
```

La sélection **Python 3.12** se fait dans **Advanced settings** avant le premier
déploiement. Si l'application a été créée en Python 3.14, supprimez uniquement
l'application Streamlit puis recréez-la ; ne supprimez pas le dépôt GitHub.

Les deux roues PyTorch CPU sont référencées directement dans `requirements.txt`.
Cette méthode évite le conflit entre PyPI et l'index PyTorch observé avec `uv`.

## 4. Test local facultatif

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python validate_deployment.py
streamlit run app.py
```

## Problèmes déjà prévenus

| Symptôme | Protection intégrée |
|---|---|
| `torch` incompatible avec Python 3.14 | Python 3.12 exigé et versions épinglées |
| Colab utilise Python 3.13 | préparation autorisée ; Python 3.12 reste réservé au déploiement |
| GitHub refuse les fichiers >25 Mo | checkpoint découpé en morceaux de 20 MiB |
| Pointeur Git LFS au lieu du poids | diagnostic explicite au démarrage |
| Morceau absent ou corrompu | manifeste + vérification SHA-256 |
| `image/jpeg files are not allowed` | validation du contenu avec Pillow, sans filtre MIME strict |
| Mauvais checkpoint | chargement `strict=True`, ViT I-JEPA officiel ou ViT timm ImageNet |
| Modèle rechargé à chaque interaction | cache pour les tests individuels |
| Mémoire insuffisante en comparaison | chargement séquentiel, un modèle à la fois |

La structure technique détaillée et les responsabilités de chaque fichier sont
décrites dans [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Avertissement

Cette application est un démonstrateur académique de perception 2D. La comparaison
qualitative sur une seule image ne remplace pas l'évaluation KITTI. La baseline
ImageNet obtient AP@50 Pedestrian 0,7949 contre 0,7261 pour Driving-Aware I-JEPA ;
la contribution I-JEPA doit donc être présentée comme l'étude d'une représentation
SSL adaptée à la conduite, et non comme une supériorité sur ImageNet. Les
catégories de proximité sont des estimations basées sur la surface apparente des
bounding boxes et non des distances métriques. Les résultats ne doivent jamais
être utilisés pour commander un véhicule.
