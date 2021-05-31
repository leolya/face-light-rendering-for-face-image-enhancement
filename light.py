import numpy as np
from utils import get_normal, render_texture
from skimage.io import imsave


def lighting(normals, colors, light_direction=[1, 1, 1], light_intensities=0.5, ambient_light_intensity=0.5):
    """
    a single directional light and ambient light
    """
    # diffuse
    light_direction_n = np.array(light_direction) / np.sqrt(np.sum(np.array(light_direction)**2))

    normals_dot_lights = normals[ :, :] * light_direction_n[np.newaxis, :]
    normals_dot_lights = np.sum(normals_dot_lights, axis = 1)
    diffuse_output = colors[ :, :] * normals_dot_lights[:, np.newaxis] * light_intensities
    diffuse_output = np.maximum(diffuse_output, 0)

    lit_colors = diffuse_output + colors[:, :] * ambient_light_intensity
    lit_colors = np.minimum(np.maximum(lit_colors, 0), 1)
    return lit_colors


def add_light_sh(vertices, triangles, colors, sh_coeff):

    assert vertices.shape[0] == colors.shape[0]
    n_ver = vertices.shape[0]
    normal = get_normal(vertices, triangles)  # (n_ver, 3)
    x = normal[:, 0]
    y = normal[:, 1]
    z = normal[:, 2]

    att = np.pi * np.array([1, 2.0 / 3.0, 1 / 4.0])

    n0 = att[0] * np.sqrt(1.0 / np.pi) / 2.0
    n1 = att[1] * np.sqrt(3.0 / np.pi) / 2.0
    n2_2 = att[2] * np.sqrt(15.0 / np.pi) / 4.0
    n2_1 = att[2] * np.sqrt(15.0 / np.pi) / 2.0
    n2_0 = att[2] * np.sqrt(5.0 / np.pi) / 4.0

    sh = np.array(
        (n0*np.ones(n_ver),
         -n1*y, n1*z, n1*x,
         -n2_1*x*y, -n2_1*y*z, n2_0*(2*z*z-x*x-y*y), n2_1*z*x, n2_2*(x*x-y*y)
         )
    ).T  # (n_ver, 9)

    ref = sh.dot(sh_coeff)  # (n_ver, 1) or (n_ver, 3)

    # white light
    ref_gray = 0.299*ref[:, 0] + 0.587*ref[:, 1] + 0.114*ref[:, 2]
    ref[:, 0] = ref_gray
    ref[:, 1] = ref_gray
    ref[:, 2] = ref_gray

    lit_colors = colors * ref  # (n_ver, 3)
    return lit_colors


def add_light_sh_normal(normal, colors, sh_coeff):

    n_ver = colors.shape[0]
    x = normal[:, 0]
    y = normal[:, 1]
    z = normal[:, 2]

    att = np.pi * np.array([1, 2.0 / 3.0, 1 / 4.0])

    n0 = att[0] * np.sqrt(1.0 / np.pi) / 2.0
    n1 = att[1] * np.sqrt(3.0 / np.pi) / 2.0
    n2_2 = att[2] * np.sqrt(15.0 / np.pi) / 4.0
    n2_1 = att[2] * np.sqrt(15.0 / np.pi) / 2.0
    n2_0 = att[2] * np.sqrt(5.0 / np.pi) / 4.0

    sh = np.array(
        (n0*np.ones(n_ver),
         -n1*y, n1*z, n1*x,
         -n2_1*x*y, -n2_1*y*z, n2_0*(2*z*z-x*x-y*y), n2_1*z*x, n2_2*(x*x-y*y)
         )
    ).T  # (n_ver, 9)

    ref = sh.dot(sh_coeff)  # (n_ver, 1) or (n_ver, 3)

    # white light
    ref_gray = 0.299*ref[:, 0] + 0.587*ref[:, 1] + 0.114*ref[:, 2]
    ref[:, 0] = ref_gray
    ref[:, 1] = ref_gray
    ref[:, 2] = ref_gray

    lit_colors = colors * ref  # (n_ver, 3)
    return lit_colors


def sphere_sh_demo(objFilePath, save_path, sh_coeff):
    h = w = 300

    with open(objFilePath) as file:
        points = []
        faces = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                points.append((float(strs[1]), float(strs[2]), float(strs[3])))
            if strs[0] == "f":
                faces.append((int(strs[3]), int(strs[2]), int(strs[1])))

    points = np.array(points)
    points = points + 150
    points[:, 1] = h - 1 - points[:, 1]
    faces = np.array(faces) - 1
    colors = np.array([[194, 142, 114]]) / 255.  # skin color sphere
    # colors = np.array([[255, 255, 255]]) / 255. # white color sphere
    colors = colors + np.zeros((2502, 3))

    # image = render_texture(points.T, colors.T, faces.T, h, w, c=3)
    # imsave(save_path + "sphere.jpg", (image*255).astype(np.uint8))

    lit_colors = add_light_sh(points, faces, colors, sh_coeff)
    lit_colors = np.clip(lit_colors, 0, 1)

    image = render_texture(points.T, lit_colors.T, faces.T, h, w, c=3)
    imsave(save_path, (image*255).astype(np.uint8))


def sphere_direct_demo(objFilePath, save_path,
                       light_direction=[1, 1, 1], light_intensities=0.5, ambient_light_intensity=0.5):
    h = w = 300

    with open(objFilePath) as file:
        points = []
        faces = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                points.append((float(strs[1]), float(strs[2]), float(strs[3])))
            if strs[0] == "f":
                faces.append((int(strs[3]), int(strs[2]), int(strs[1])))

    points = np.array(points)
    points = points + 150
    points[:, 1] = h - 1 - points[:, 1]
    faces = np.array(faces) - 1

    colors = np.array([[194, 142, 114]]) / 255.  # skin color sphere
    colors = colors + np.zeros((2502, 3))

    normals = get_normal(points, faces)
    lit_colors = lighting(normals, colors, light_direction=light_direction,
                    light_intensities=light_intensities, ambient_light_intensity=ambient_light_intensity)
    image = render_texture(points.T, lit_colors.T, faces.T, h, w, c=3)
    image = np.reshape(image, (h, w, 3))
    imsave(save_path, (image*255).astype(np.uint8))



