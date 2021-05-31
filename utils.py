import numpy as np


def get_normal(vertices, triangles):
    ''' calculate normal direction in each vertex
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3]
    Returns:
        normal: [nver, 3] (range -1 to 1)
    '''
    pt0 = vertices[triangles[:, 0], :]  # [ntri, 3]
    pt1 = vertices[triangles[:, 1], :]  # [ntri, 3]
    pt2 = vertices[triangles[:, 2], :]  # [ntri, 3]
    tri_normal = np.cross(pt0 - pt1, pt0 - pt2)  # [ntri, 3]. normal of each triangle

    normal = np.zeros_like(vertices)  # [nver, 3]
    for i in range(triangles.shape[0]):
        # 顶点的normal是所有连接的面的normal的和
        normal[triangles[i, 0], :] = normal[triangles[i, 0], :] + tri_normal[i, :]
        normal[triangles[i, 1], :] = normal[triangles[i, 1], :] + tri_normal[i, :]
        normal[triangles[i, 2], :] = normal[triangles[i, 2], :] + tri_normal[i, :]

    # normalize to unit length
    mag = np.sum(normal ** 2, 1)  # [nver]
    zero_ind = (mag == 0)
    mag[zero_ind] = 1
    normal[zero_ind, 0] = np.ones((np.sum(zero_ind)))

    normal = normal / np.sqrt(mag[:, np.newaxis])

    return normal


def render_texture(vertices, colors, triangles, h, w, c=3):
    image = np.zeros((h, w, c))

    depth_buffer = np.zeros([h, w]) - 999999.
    tri_depth = (vertices[2, triangles[0, :]] + vertices[2, triangles[1, :]] + vertices[2, triangles[2, :]]) / 3.
    tri_tex = (colors[:, triangles[0, :]] + colors[:, triangles[1, :]] + colors[:, triangles[2, :]]) / 3.

    for i in range(triangles.shape[1]):
        tri = triangles[:, i]  # 3 vertex indices

        # the inner bounding box
        umin = max(int(np.ceil(np.min(vertices[0, tri]))), 0)
        umax = min(int(np.floor(np.max(vertices[0, tri]))), w - 1)

        vmin = max(int(np.ceil(np.min(vertices[1, tri]))), 0)
        vmax = min(int(np.floor(np.max(vertices[1, tri]))), h - 1)

        if umax < umin or vmax < vmin:
            continue

        for u in range(umin, umax + 1):
            for v in range(vmin, vmax + 1):
                if tri_depth[i] > depth_buffer[v, u] and isPointInTri([u, v], vertices[:2, tri]):
                    depth_buffer[v, u] = tri_depth[i]
                    image[v, u, :] = tri_tex[:, i]
    return image


def isPointInTri(point, tri_points):
    ''' Judge whether the point is in the triangle
    Method:
        http://blackpawn.com/texts/pointinpoly/
    Args:
        point: [u, v] or [x, y]
        tri_points: three vertices(2d points) of a triangle. 2 coords x 3 vertices
    Returns:
        bool: true for in triangle
    '''
    tp = tri_points

    # vectors
    v0 = tp[:, 2] - tp[:, 0]
    v1 = tp[:, 1] - tp[:, 0]
    v2 = point - tp[:, 0]

    # dot products
    dot00 = np.dot(v0.T, v0)
    dot01 = np.dot(v0.T, v1)
    dot02 = np.dot(v0.T, v2)
    dot11 = np.dot(v1.T, v1)
    dot12 = np.dot(v1.T, v2)

    # barycentric coordinates
    if dot00 * dot11 - dot01 * dot01 == 0:
        inverDeno = 0
    else:
        inverDeno = 1 / (dot00 * dot11 - dot01 * dot01)

    u = (dot11 * dot02 - dot01 * dot12) * inverDeno
    v = (dot00 * dot12 - dot01 * dot02) * inverDeno

    # check if point in triangle
    return (u >= 0) & (v >= 0) & (u + v < 1)