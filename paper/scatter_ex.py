import numpy as np
from mayavi import mlab

np.random.seed(42)

mlab.figure(bgcolor=(1,1,1))
xyz = np.random.random((10, 3))

colors = np.repeat([1,2], xyz.shape[0]/2)
pt = mlab.points3d(xyz[:,0],xyz[:,1], xyz[:,2], colors, scale_factor=.1)

pt.glyph.color_mode='color_by_scalar'

cmap = np.array([[255, 0,   0, 128],
                 [0,   255, 0, 255]])
cmap = np.tile(cmap, (int(xyz.shape[0]/2), 1))

pt.module_manager.scalar_lut_manager.lut._vtk_obj.SetTableRange(1, cmap.shape[0])
pt.module_manager.scalar_lut_manager.lut.number_of_colors = cmap.shape[0]
pt.module_manager.scalar_lut_manager.lut.table = cmap
# pt.update_pipeline()

mlab.draw()
mlab.show()


