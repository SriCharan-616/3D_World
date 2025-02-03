import numpy as np
import moderngl as mgl
import pywavefront
import pygame.freetype
import random
import string
import numpy as np
import glm

class VBO:
    def __init__(self, ctx):
        self.vbos = {}
        self.vbos['cube'] = CubeVBO(ctx)
        self.vbos['cat'] = CatVBO(ctx)
        self.vbos['skybox'] = SkyBoxVBO(ctx)
        self.vbos['advanced_skybox'] = AdvancedSkyBoxVBO(ctx)
        self.vbos['box'] = BoxVBO(ctx)  # Add the Box VBO

    def destroy(self):
        [vbo.destroy() for vbo in self.vbos.values()]


class BaseVBO:
    def __init__(self, ctx):
        self.ctx = ctx
        self.vbo = self.get_vbo()
        self.format: str = None
        self.attribs: list = None

    def get_vertex_data(self): ...

    def get_vbo(self):
        vertex_data = self.get_vertex_data()
        vbo = self.ctx.buffer(vertex_data)
        return vbo

    def destroy(self):
        self.vbo.release()


class CubeVBO(BaseVBO):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.format = '2f 3f 3f'
        self.attribs = ['in_texcoord_0', 'in_normal', 'in_position']

    @staticmethod
    def get_data(vertices, indices):
        data = [vertices[ind] for triangle in indices for ind in triangle]
        return np.array(data, dtype='f4')

    def get_vertex_data(self):
        vertices = [(-1, -1, 1), ( 1, -1,  1), (1,  1,  1), (-1, 1,  1),
                    (-1, 1, -1), (-1, -1, -1), (1, -1, -1), ( 1, 1, -1)]

        indices = [(0, 2, 3), (0, 1, 2),
                   (1, 7, 2), (1, 6, 7),
                   (6, 5, 4), (4, 7, 6),
                   (3, 4, 5), (3, 5, 0),
                   (3, 7, 4), (3, 2, 7),
                   (0, 6, 1), (0, 5, 6)]
        vertex_data = self.get_data(vertices, indices)

        tex_coord_vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
        tex_coord_indices = [(0, 2, 3), (0, 1, 2),
                             (0, 2, 3), (0, 1, 2),
                             (0, 1, 2), (2, 3, 0),
                             (2, 3, 0), (2, 0, 1),
                             (0, 2, 3), (0, 1, 2),
                             (3, 1, 2), (3, 0, 1),]
        tex_coord_data = self.get_data(tex_coord_vertices, tex_coord_indices)

        normals = [( 0, 0, 1) * 6,
                   ( 1, 0, 0) * 6,
                   ( 0, 0,-1) * 6,
                   (-1, 0, 0) * 6,
                   ( 0, 1, 0) * 6,
                   ( 0,-1, 0) * 6,]
        normals = np.array(normals, dtype='f4').reshape(36, 3)

        vertex_data = np.hstack([normals, vertex_data])
        vertex_data = np.hstack([tex_coord_data, vertex_data])
        return vertex_data


class CatVBO(BaseVBO):
    def __init__(self, app):
        super().__init__(app)
        self.format = '2f 3f 3f'
        self.attribs = ['in_texcoord_0', 'in_normal', 'in_position']

    def get_vertex_data(self):
        objs = pywavefront.Wavefront('objects/cat/20430_Cat_v1_NEW.obj', cache=True, parse=True)
        obj = objs.materials.popitem()[1]
        vertex_data = obj.vertices
        vertex_data = np.array(vertex_data, dtype='f4')
        return vertex_data


class SkyBoxVBO(BaseVBO):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.format = '3f'
        self.attribs = ['in_position']

    @staticmethod
    def get_data(vertices, indices):
        data = [vertices[ind] for triangle in indices for ind in triangle]
        return np.array(data, dtype='f4')

    def get_vertex_data(self):
        vertices = [(-1, -1, 1), ( 1, -1,  1), (1,  1,  1), (-1, 1,  1),
                    (-1, 1, -1), (-1, -1, -1), (1, -1, -1), ( 1, 1, -1)]

        indices = [(0, 2, 3), (0, 1, 2),
                   (1, 7, 2), (1, 6, 7),
                   (6, 5, 4), (4, 7, 6),
                   (3, 4, 5), (3, 5, 0),
                   (3, 7, 4), (3, 2, 7),
                   (0, 6, 1), (0, 5, 6)]
        vertex_data = self.get_data(vertices, indices)
        vertex_data = np.flip(vertex_data, 1).copy(order='C')
        return vertex_data


class AdvancedSkyBoxVBO(BaseVBO):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.format = '3f'
        self.attribs = ['in_position']

    def get_vertex_data(self):
        # in clip space
        z = 0.9999
        vertices = [(-1, -1, z), (3, -1, z), (-1, 3, z)]
        vertex_data = np.array(vertices, dtype='f4')
        return vertex_data
#My custom box  
#My custom box  
class BoxVBO(BaseVBO):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.format = '2f 3f 3f'
        self.attribs = ['in_texcoord_0', 'in_normal', 'in_position']
        self.font = pygame.freetype.SysFont("Arial", 64)
        self.letter = self.get_random_letter()
        self.last_update = pygame.time.get_ticks()
        self.update_interval = 2000  # Update every 2 seconds
        self.texture = self.create_letter_texture(self.letter)

    @staticmethod
    def get_data(vertices, indices):
        # Ensure indices are within bounds of the vertices list
        try:
            data = [vertices[ind] for triangle in indices for ind in triangle]
        except IndexError:
            print("Index out of range!")
            raise
        return np.array(data, dtype='f4')

    def get_vertex_data(self):
        vertices = [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
                    (-1, 1, -1), (-1, -1, -1), (1, -1, -1), (1, 1, -1)]

        indices = [(0, 2, 3), (0, 1, 2),
                   (1, 7, 2), (1, 6, 7),
                   (6, 5, 4), (4, 7, 6),
                   (3, 4, 5), (3, 5, 0),
                   (3, 7, 4), (3, 2, 7),
                   (0, 6, 1), (0, 5, 6)]
        
        # Ensure indices are valid
        try:
            vertex_data = self.get_data(vertices, indices)
        except IndexError:
            print("Index out of range in vertex data!")
            raise

        tex_coord_vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
        tex_coord_indices = [(0, 2, 3), (0, 1, 2),
                             (0, 2, 3), (0, 1, 2),
                             (0, 1, 2), (2, 3, 0),
                             (2, 3, 0), (2, 0, 1),
                             (0, 2, 3), (0, 1, 2),
                             (3, 1, 2), (3, 0, 1),]

        try:
            tex_coord_data = self.get_data(tex_coord_vertices, tex_coord_indices)
        except IndexError:
            print("Index out of range in tex_coord data!")
            raise

        normals = [(0, 0, 1) * 6,
                   (1, 0, 0) * 6,
                   (0, 0, -1) * 6,
                   (-1, 0, 0) * 6,
                   (0, 1, 0) * 6,
                   (0, -1, 0) * 6,]
        normals = np.array(normals, dtype='f4').reshape(36, 3)

        vertex_data = np.hstack([normals, vertex_data])
        vertex_data = np.hstack([tex_coord_data, vertex_data])
        return vertex_data

    def get_random_letter(self):
        return random.choice(string.ascii_uppercase)

    def create_letter_texture(self, letter):
        surface = pygame.Surface((256, 256), pygame.SRCALPHA)
        surface.fill((0, 0, 0, 255))  # Black background
        self.font.render_to(surface, (80, 100), letter, (255, 255, 255))  # White letter
        texture_data = pygame.image.tostring(surface, "RGBA", True)
        
        texture = self.ctx.texture((256, 256), 4, texture_data)
        texture.repeat_x = False
        texture.repeat_y = False
        texture.build_mipmaps()
        return texture

    def update(self):
        current_time = pygame.time.get_ticks()
        if current_time - self.last_update > self.update_interval:
            self.letter = self.get_random_letter()
            self.texture = self.create_letter_texture(self.letter)
            self.last_update = current_time 
    
    def use_texture(self):
        self.texture.use(location=0)