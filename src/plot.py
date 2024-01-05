from PIL import Image

def openImage(path):
    return Image.open(path)

def addIconInImage(image, icon, position, dim_icon=35):
    # Charger l'image principale
    image = image.convert("RGBA")

    # Charger l'image à insérer
    icon = icon.convert("RGBA").resize((dim_icon, dim_icon))

    # Définir les coordonnées de l'endroit où vous souhaitez insérer l'image
    position_x = position[0]
    position_y = position[1]
    position_x = int(position_x)
    position_y = int(position_y)

    # Insérer l'icon à la position spécifiée
    image.paste(icon, (position_x, position_y), mask=icon)

    # Afficher ou enregistrer l'image modifiée
    return image

def plot_access(
    access_points,
    path_batiment,
    path_icon,
    dim_icon=35,
    echelle=100,
    path_computed=None,
    showed=True,
    image = None
):
    image_batiment = Image.open(path_batiment) if image is None else image.copy()
    icon = Image.open(path_icon)

    # ajout des points d'accès dans le batiment
    image = image_batiment.copy()
    for pos in access_points.values():
        position = [echelle * pos[0], echelle * pos[1]]
        image = addIconInImage(image, icon, position, dim_icon=dim_icon)

    if showed:
        image.show()
    if path_computed is not None:
        image.save(path_computed, format="png")

    return image


def plot_position(
    M,
    path_batiment,
    path_icon,
    dim_icon=35,
    echelle=100,
    path_computed=None,
    showed=True,
    image = None
):
    image_batiment = Image.open(path_batiment) if image is None else image.copy()
    icon = Image.open(path_icon)

    # ajout de la position dans le batiment
    position = [echelle * M[0], echelle * M[1]]
    image = addIconInImage(image_batiment, icon, position, dim_icon=dim_icon)

    if showed:
        image.show()
    if path_computed is not None:
        image.save(path_computed, format="png")

    return image


def plot_access_user(
    access_points,
    M,
    path_batiment,
    path_icon_access,
    path_icon_position,
    dim_icon=35,
    echelle=100,
    path_computed=None,
    showed=True,
    image = None
):
    image_batiment = Image.open(path_batiment) if image is None else image.copy()
    icon_access = Image.open(path_icon_access)
    icon_position = Image.open(path_icon_position)

    # ajout des points d'accès dans le batiment
    image = image_batiment.copy()
    for pos in access_points.values():
        position = [echelle * pos[0], echelle * pos[1]]
        image = addIconInImage(image, icon_access, position, dim_icon=dim_icon)

    # ajout de la position dans le batiment
    position = [echelle * M[0], echelle * M[1]]
    image = addIconInImage(image, icon_position, position)

    if showed:
        image.show()
    if path_computed is not None:
        image.save(path_computed, format="png")

    return image
