"""
Visualization utilities for TransLNP project.
Provides functions for plotting molecules with their properties in a spiral layout.
"""

from typing import Iterable, List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage, TextArea, VPacker
from PIL import Image
import io
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

def trim_image(image: Image.Image, padding: int = 5) -> Image.Image:
    """Trim whitespace from an image while preserving padding.
    
    Args:
        image: PIL Image to trim
        padding: Number of pixels to preserve around the content
        
    Returns:
        Trimmed PIL Image
    """
    bbox = image.getbbox()
    if bbox:
        return image.crop(bbox).resize((image.width, image.height))
    return image

def _nearest_spiral_layout(x: np.ndarray, y: np.ndarray, offset: int) -> np.ndarray:
    """Generate spiral layout coordinates for molecule placement.
    
    Args:
        x: X coordinates of points
        y: Y coordinates of points
        offset: Offset in spiral layout
        
    Returns:
        Array of spiral coordinates
    """
    angles = np.linspace(-np.pi, np.pi, len(x) + 1 + offset)[offset:]
    coords = np.stack((np.cos(angles), np.sin(angles)), -1)
    order = np.argsort(np.arctan2(y, x))
    return coords[order]

def _image_scatter(
    x: np.ndarray,
    y: np.ndarray,
    imgs: List[Image.Image],
    subtitles: List[str],
    colors: List[str],
    ax: plt.Axes,
    offset: Union[int, Tuple[float, float]] = 0
) -> List[AnnotationBbox]:
    """Create scatter plot with molecule images and subtitles.
    
    Args:
        x: X coordinates
        y: Y coordinates
        imgs: List of molecule images
        subtitles: List of subtitles for each molecule
        colors: List of colors for each molecule box
        ax: Matplotlib axes to plot on
        offset: Offset for image placement (int for spiral, tuple for fixed)
        
    Returns:
        List of annotation boxes
    """
    if isinstance(offset, int):
        box_coords = _nearest_spiral_layout(x, y, offset)
    elif isinstance(offset, Iterable) and len(offset) == 2:
        box_coords = offset
    else:
        raise ValueError("offset must be int or tuple of length 2")
        
    bbs = []
    for i, (x0, y0, im, t, c) in enumerate(zip(x, y, imgs, subtitles, colors)):
        # Trim whitespace from image
        im = trim_image(im)
        img_data = np.asarray(im)
        
        # Create image box with unique ID
        img_box = OffsetImage(img_data, zoom=1.0)
        title_box = TextArea(t)
        packed = VPacker(children=[img_box, title_box], pad=0, sep=4, align="center")
        
        # Create annotation box
        bb = AnnotationBbox(
            packed,
            (x0, y0),
            frameon=True,
            xybox=box_coords[i] + 0.5 if isinstance(offset, int) else offset,
            arrowprops=dict(arrowstyle="->", edgecolor="black"),
            pad=0.3,
            boxcoords="axes fraction",
            bboxprops=dict(edgecolor=c),
        )
        ax.add_artist(bb)
        bbs.append(bb)
    return bbs

def plot_molecules(
    molecules: List[Chem.Mol],
    x: np.ndarray,
    y: np.ndarray,
    subtitles: List[str],
    colors: List[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    offset: Union[int, Tuple[float, float]] = 0,
    title: str = None
) -> plt.Figure:
    """Create a plot of molecules with their properties.
    
    Args:
        molecules: List of RDKit molecules
        x: X coordinates for each molecule
        y: Y coordinates for each molecule
        subtitles: List of subtitles for each molecule
        colors: List of colors for each molecule box (default: use colormap)
        figsize: Figure size
        offset: Offset for image placement
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Generate molecule images
    imgs = []
    for mol in molecules:
        # Generate 2D coordinates if needed
        if mol.GetNumConformers() == 0:
            AllChem.Compute2DCoords(mol)
        
        # Draw molecule
        img = Draw.MolToImage(mol, size=(300, 300))
        imgs.append(img)
    
    # Use default colors if none provided
    if colors is None:
        colors = facecolors_customize[:len(molecules)]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    _image_scatter(x, y, imgs, subtitles, colors, ax, offset)
    
    if title:
        ax.set_title(title)
    
    return fig

# Custom colormap for molecule visualization
facecolors_customize = [
    "#a6d9daff", "#96c9ccff", "#91bfc2ff", "#8cb7baff", "#8aafb2ff",
    "#87a5aaff", "#849d9fff", "#829495ff", "#829091ff", "#828c8bff",
    "#828486ff", "#828282ff", "#807a7dff", "#79727aff", "#726674ff",
    "#6a596fff", "#624e67ff", "#5a4261ff", "#513957ff", "#492f4fff",
    "#40263eff", "#39233fff", "#32203fff",
]

def main():
    """Example usage of the plotting functions."""
    # Create some example molecules
    smiles_list = ["CC(=O)OC1=CC=CC=C1C(=O)O", "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(N)(=O)=O)C(F)(F)F"]
    molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    
    # Example coordinates and subtitles
    x = np.array([0, 1])
    y = np.array([0, 1])
    subtitles = ["Aspirin", "Celecoxib"]
    
    # Create plot
    fig = plot_molecules(
        molecules=molecules,
        x=x,
        y=y,
        subtitles=subtitles,
        title="Example Molecule Plot"
    )
    
    plt.show()

if __name__ == "__main__":
    main()