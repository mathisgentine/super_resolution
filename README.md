Code related to MSc Dissertation : "Deep learning microscopy enhancement: from low to high-resolution imaging"

Data can be found here : https://www.dropbox.com/sh/24lym7r4h2uejb2/AAC_jp9W64d0hgwN6Z34_x6ea?dl=0

----

Most of the research work has been done on `Dissertation - Colab.ipynb`. All functions mentioned in the write-up can be found there.

`image_slicer_tool.py` and `patchify_tool.py` provide a way to respectively perform the **slicing** and **patchify** splitting explained in part 3.4.

To get a prediction a full image, use either `predict.py` or `predict_patchify.py`.
One simply needs to create a folder named **`{zoom_type}-{number_of_tiles}`** (where "zoom_type" refers to either **4x20x** or **20x100x**) and place an `export.pkl` file inside (obtained by running `learn.export()` on a model `learn`). 
We provide two export files from the training of a **U-Net - Classic - Patchify** model.

**4x20x-100** export : https://www.dropbox.com/sh/ni222ln43ngw24c/AAApJpnegw60DIE2d3DuYEWfa?dl=0 <br/>
**20x100x-100** export : https://www.dropbox.com/sh/hrw82v9rdf34wsk/AACXqEBX5m6ZBcwaWqWG9ZKna?dl=0


It takes respectively 2:43 & 9:55 minutes to predict the output of one full image, on my CPU (2,7 GHz Intel Core i5).

Outputs on a **20x100x** test image are available to download. 
We show a reduced, png-version comparison below, where the left image was generated with `predict.py` and the right one with `predict_patchify.py` (with the same `export.pkl` file):

![Comparison](https://github.com/mathisgentine/super_resolution/blob/master/Output_comparison.png)
