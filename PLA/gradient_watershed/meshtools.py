#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 13:25:24 2018

@author: felix

"""

def read_mesh(meshfile, 
              process=False, 
              validate=False, 
              keep_largest_only=False):

    import trimesh 
    import numpy as np 

    mesh = trimesh.load_mesh(meshfile, 
                             validate=validate, 
                             process=process)
    if keep_largest_only:
        mesh_comps = mesh.split(only_watertight=False)
        mesh = mesh_comps[np.argmax([len(cc.vertices) for cc in mesh_comps])]

    return mesh 

def create_mesh(vertices,faces,vertex_colors=None, face_colors=None):

    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices,
                            faces=faces, 
                            process=False,
                            validate=False, 
                            vertex_colors=vertex_colors, 
                            face_colors=face_colors)

    return mesh 

def submesh(mesh,
            faces_sequence,
            mesh_face_attributes,
            repair=True,
            only_watertight=False,
            min_faces=None,
            append=False,**kwargs):
        
    """
    Return a subset of a mesh.
    Parameters
    ------------
    mesh : Trimesh
       Source mesh to take geometry from
    faces_sequence : sequence (p,) int
        Indexes of mesh.faces
    only_watertight : bool
        Only return submeshes which are watertight.
    append : bool
        Return a single mesh which has the faces appended,
        if this flag is set, only_watertight is ignored
    Returns
    ---------
    if append : Trimesh object
    else        list of Trimesh objects
    """
    import copy
    import numpy as np

    def type_bases(obj, depth=4):
        """
        Return the bases of the object passed.
        """
        import collections
        bases = collections.deque([list(obj.__class__.__bases__)])
        for i in range(depth):
            bases.append([i.__base__ for i in bases[-1] if i is not None])
        try:
            bases = np.hstack(bases)
        except IndexError:
            bases = []
        # we do the hasattr as None/NoneType can be in the list of bases
        bases = [i for i in bases if hasattr(i, '__name__')]
        return np.array(bases)

    
    def type_named(obj, name):
        """
        Similar to the type() builtin, but looks in class bases
        for named instance.
        Parameters
        ------------
        obj: object to look for class of
        name : str, name of class
        Returns
        ----------
        named class, or None
        """
        # if obj is a member of the named class, return True
        name = str(name)
        if obj.__class__.__name__ == name:
            return obj.__class__
        for base in type_bases(obj):
            if base.__name__ == name:
                return base
        raise ValueError('Unable to extract class of name ' + name)
    
    # evaluate generators so we can escape early
    faces_sequence = list(faces_sequence)

    if len(faces_sequence) == 0:
        return []

    # avoid nuking the cache on the original mesh
    original_faces = mesh.faces.view(np.ndarray)
    original_vertices = mesh.vertices.view(np.ndarray)

    faces = []
    vertices = []
    normals = []
    visuals = []
    attributes = []

    # for reindexing faces
    mask = np.arange(len(original_vertices))

    for index in faces_sequence:
        # sanitize indices in case they are coming in as a set or tuple
        index = np.asanyarray(index)
        if len(index) == 0:
            # regardless of type empty arrays are useless
            continue
        if index.dtype.kind == 'b':
            # if passed a bool with no true continue
            if not index.any():
                continue
            # if fewer faces than minimum
            if min_faces is not None and index.sum() < min_faces:
                continue
        elif min_faces is not None and len(index) < min_faces:
            continue

        current = original_faces[index]
        unique = np.unique(current.reshape(-1)) # unique points. 

        # redefine face indices from zero
        mask[unique] = np.arange(len(unique))
        normals.append(mesh.face_normals[index])
        faces.append(mask[current])
        vertices.append(original_vertices[unique])
        attributes.append(mesh_face_attributes[index])
        visuals.append(mesh.visual.face_subset(index))

    if len(vertices) == 0:
        return np.array([])

    # we use type(mesh) rather than importing Trimesh from base
    # to avoid a circular import
    trimesh_type = type_named(mesh, 'Trimesh')

    # generate a list of Trimesh objects
    result = [trimesh_type(
        vertices=v,
        faces=f,
        face_normals=n,
        visual=c,
        metadata=copy.deepcopy(mesh.metadata),
        process=False) for v, f, n, c in zip(vertices,
                                             faces,
                                             normals,
                                             visuals)]
    result = np.array(result)

    return result, attributes


def split_mesh(mesh,
                mesh_face_attributes, 
                adjacency=None, 
                only_watertight=False, 
                engine=None, **kwargs):
    
    """
    Split a mesh into multiple meshes from face
    connectivity.
    If only_watertight is true it will only return
    watertight meshes and will attempt to repair
    single triangle or quad holes.
    Parameters
    ----------
    mesh : trimesh.Trimesh
    only_watertight: bool
      Only return watertight components
    adjacency : (n, 2) int
      Face adjacency to override full mesh
    engine : str or None
      Which graph engine to use
    Returns
    ----------
    meshes : (m,) trimesh.Trimesh
      Results of splitting
    meshes_att : (m,d) attributes. 
      associated splitted attributes. 
    """
    import trimesh 
    import numpy as np 
    
    # used instead of trimesh functions in order to keep it consistent with the splitting of mesh attributes. 
    if adjacency is None:
        adjacency = mesh.face_adjacency

    # if only watertight the shortest thing we can split has 3 triangles
    if only_watertight:
        min_len = 4
    else:
        min_len = 1

    components = trimesh.graph.connected_components(
        edges=adjacency,
        nodes=np.arange(len(mesh.faces)),
        min_len=min_len,
        engine=engine)

    meshes, meshes_attributes = submesh(mesh,
                                    components, 
                                    mesh_face_attributes,
                                    **kwargs)
    return meshes, meshes_attributes



def decimate_resample_mesh(mesh, remesh_samples, predecimate=True):

    # this will for sure change the connectivity 
    import pyacvd
    import pyvista as pv
    import igl
    import trimesh

    if predecimate:
        _, V, F, _, _ = igl.decimate(mesh.vertices, mesh.faces, int(.9*len(mesh.faces))) # there is bug? 
        if len(V) > 0: # have a check in here to prevent break down. 
            mesh = trimesh.Trimesh(V, F, validate=True) # why no good? 

    # print(len(mesh.vertices))
    mesh = pv.wrap(mesh) # convert to pyvista format. 
    clus = pyacvd.Clustering(mesh)
    clus.cluster(int(remesh_samples*len(mesh.points))) # this guarantees a remesh is possible. 
    mesh = clus.create_mesh()

    mesh = trimesh.Trimesh(mesh.points, mesh.faces.reshape((-1,4))[:, 1:4], validate=True) # we don't care. if change
    # print(mesh.is_watertight)
    return mesh


def upsample_mesh(mesh, method='inplane'):

    """
    inplane or loop
    """
    import igl 
    import trimesh 

    if method =='inplane': 
        uv, uf = igl.upsample(mesh.vertices, mesh.faces)
    if method == 'loop':
        uv, uf = igl.loop(mesh.vertices, mesh.faces)
   
    mesh_out = trimesh.Trimesh(uv, uf, validate=False, process=False)
    
    return mesh_out 


def upsample_mesh_and_vertex_vals(mesh, vals, method='inplane'):

    """
    inplane only... vals is the same length as mesh vertices. 
    """
    import igl 
    import trimesh 
    import numpy as np 

    if method =='inplane': 
        uv, uf = igl.upsample(mesh.vertices, mesh.faces) # get the new vertex and faces. 
    if method == 'loop':
        uv, uf = igl.loop(mesh.vertices, mesh.faces)
    vals_new = np.zeros((len(uv), vals.shape[-1])); vals_new[:] = np.nan
    max_ind_mesh_in = len(mesh.vertices)
    vals_new[:max_ind_mesh_in] = vals.copy()

    old_new_edge_list = igl.edges(uf) # use the new faces. 
    old_new_edge_list = old_new_edge_list[old_new_edge_list[:,0]<max_ind_mesh_in]

    vals_new[old_new_edge_list[::2,1]] = .5*(vals_new[old_new_edge_list[::2,0]]+vals_new[old_new_edge_list[1::2,0]])
    mesh_out = trimesh.Trimesh(uv, uf, validate=False, process=False)
    
    return mesh_out, vals_new


def marching_cubes_mesh_binary(vol, presmooth=1., contourlevel=.5, remesh=False, 
                               remesh_method='pyacvd', remesh_samples=.5, remesh_params=None, 
                               predecimate=True, min_mesh_size=40000, keep_largest_only=True, min_comp_size=20, split_mesh=True, upsamplemethod='inplane'):

    from skimage.filters import gaussian
    import trimesh
    try:
        from skimage.measure import marching_cubes_lewiner
    except:
        from skimage.measure import marching_cubes
    import igl 
    import numpy as np 

    if presmooth is not None:
        img = gaussian(vol, sigma=presmooth, preserve_range=True)
        img = img / img.max() # do this. 
    else:
        img = vol.copy()
        
    try:
        V, F, _, _ = marching_cubes_lewiner(img, level=contourlevel, allow_degenerate=False)
    except:
        V, F, _, _ = marching_cubes(img, level=contourlevel, method='lewiner')
    mesh = trimesh.Trimesh(V,F, validate=True)
    
    if split_mesh:
        mesh_comps = mesh.split(only_watertight=False)
        
        if keep_largest_only:
            mesh = mesh_comps[np.argmax([len(cc.vertices) for cc in mesh_comps])]
        else:
            mesh_comps = [mm for mm in mesh_comps if len(mm.faces)>=min_comp_size] # keep a min_size else the remeshing doesn't work 
            # combine_mesh_components
            mesh = trimesh.util.concatenate(mesh_comps)
        # we need to recombine this
        # mesh = mesh_comps[np.argmax([len(cc.vertices) for cc in mesh_comps])]
    if remesh:
        if remesh_method == 'pyacvd':
            mesh = decimate_resample_mesh(mesh, remesh_samples, predecimate=predecimate)
            # other remesh is optimesh which allows us to reshift the vertices (change the connections)
        if remesh_method == 'optimesh':
            if predecimate:
                _, V, F, _, _ = igl.decimate(mesh.vertices,mesh.faces, int(.9*len(mesh.faces))) # decimates up to the desired amount of faces? 
                mesh = trimesh.Trimesh(V, F, validate=True)
            mesh, _, mean_quality = relax_mesh( mesh, relax_method=remesh_params['relax_method'], tol=remesh_params['tol'], n_iters=remesh_params['n_iters']) # don't need the quality parameters. 
            print('mean mesh quality: ', mean_quality)

    mesh_check = len(mesh.vertices) >= min_mesh_size # mesh_min size is only applied here.!
    # while(mesh_check==0):
    if mesh_check==0:
        mesh = upsample_mesh(mesh, method=upsamplemethod)
        mesh_check = len(mesh.vertices) >= min_mesh_size
        print(mesh_check)

    return mesh


def combine_submeshes(mesh_list):
    
    import trimesh
    
    combmesh = trimesh.util.concatenate(mesh_list) # why is there an outlier at the border? 
    
    return combmesh