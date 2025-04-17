# -*- coding: utf-8 -*-
"""
Created on Thu 6/6/2024
@author: Tada
    Strcuture: rootfolder, subfolder 
    Assumes rootfolders contains subfolders, consisting of cell lines or conditions 
    Define conditions and experimental folders at the beginning
    
    
    Analyses folder will be saved in analysisRoot
    savefolders will contain all input, mask_overlay and segmantion.tiff 
"""

# """
# Define rootfolder (containing subfolders with cell lines or conditions)
# """
def mkdir(directory):
    """ check if directory exists and create it through Python if it does not yet.

    Parameters
    ----------
    directory : str
        the directory path (absolute or relative) you wish to create (or check that it exists)

    Returns
    -------
        void function, no return
    """
    import os

    if not os.path.exists(directory):
        os.makedirs(directory)

    return []

import os
import numpy as np

#Define condition glass or soft
condition = 'soft'

#Define which experiment to analyze
exp = '240520_exp2'

#Where is the Raw data
rootfolder = os.path.join('/project/bioinformatics/Danuser_lab/3Dmorphogenesis/raw/tisogai/TIRF-2', exp+'_U2OS_VCLARPC2-PLA', condition) #Change this accordingly

#What is the root address where you would like to store the analyses?
analysisRoot ='/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/tisogai/3D-ARP3 Project/PLA_VCL-ARPC2'

#Save the analysis folder according to the experiment
analysisRoot = os.path.join(analysisRoot, exp)
mkdir(analysisRoot)

# """
# Specify the rescale size (20-40px of target object)
# """
factor = 0.025 #use around 0.02-0.03 for glass and 0.04-0.05 for soft

# """
# This script uses ch_VCL for the segmentation
# """
ch_VCL = 0
ch_Factin = 1
# ch_3 = 2

def imadjust(vol, p1, p2):
    import numpy as np
    from skimage.exposure import rescale_intensity
    # this is based on contrast stretching and is used by many of the biological image processing algorithms.
    p1_, p2_ = np.percentile(vol, (p1,p2))
    vol_rescale = rescale_intensity(vol, in_range=(p1_,p2_))
    return vol_rescale

def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x   = x.astype(dtype,copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
        eps = dtype(eps)
    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x =                   (x - mi) / ( ma - mi + eps )
    if clip:
        x = np.clip(x,0,1)
    return x

def normalize(x, pmin=2, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""

    mi = np.percentile(x,pmin,axis=axis,keepdims=True)
    ma = np.percentile(x,pmax,axis=axis,keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)

def distance_transform_labels(labels, bg_label=0):

    import numpy as np
    from scipy.ndimage import distance_transform_edt
    # import skfmm

    dtform = np.zeros(labels.shape)

    uniq_labels = np.setdiff1d(np.unique(labels), bg_label)

    for lab in uniq_labels:

        mask = labels == lab
        # dist_mask = skfmm.distance(mask>0)
        dist_mask = distance_transform_edt(mask>0)
        dtform[mask>0] = dist_mask[mask>0]

    return dtform

# =============================================================================
#  Use the euclidean distance transform
# =============================================================================
def distance_transform_labels_fast(labels, n_threads=16, black_border=False):
    """ compute euclidean distance transform for each uniquely labelled cell.

    """
    import numpy as np
    # from scipy.ndimage import distance_transform_edt
    # import skfmm
    import edt

    dtform = np.array([edt.edt(ss, black_border=black_border,
                               order='C',
                               parallel=n_threads).astype(np.float32) for ss in labels])

    return dtform

def sdf_distance_transform(binary, rev_sign=True):

    import numpy as np
    from scipy.ndimage import distance_transform_edt
    # import skfmm
    # import GeodisTK

    pos_binary = binary.copy()
    neg_binary = np.logical_not(pos_binary)

    res = distance_transform_edt(neg_binary) * neg_binary - (distance_transform_edt(pos_binary) - 1) * pos_binary
    # res = skfmm.distance(neg_binary, dx=0.5) * neg_binary - (skfmm.distance(pos_binary, dx=0.5) - 1) * pos_binary
    # res = skfmm.distance(neg_binary) * neg_binary - (skfmm.distance(pos_binary) - 1) * pos_binary # this was fast!.
    # res = geodesic_distance_2d((neg_binary*1.).astype(np.float32), S=neg_binary, lamb=0.8, iter=10) * neg_binary - (geodesic_distance_2d((pos_binary*1.).astype(np.float32), S=neg_binary, lamb=0.5, iter=10) - 1) * pos_binary

    if rev_sign:
        res = res * -1

    return res

def surf_normal_sdf(binary, return_sdf=True, smooth_gradient=None, eps=1e-12, norm_vectors=True):

    import numpy as np
    import scipy.ndimage as ndimage

    sdf_vol = sdf_distance_transform(binary, rev_sign=True) # so that we have it pointing outwards!.

    # compute surface normal of the signed distance function.
    sdf_vol_normal = np.array(np.gradient(sdf_vol))
    # smooth gradient
    if smooth_gradient is not None: # smoothing needs to be done before normalization of magnitude.
        sdf_vol_normal = np.array([ndimage.gaussian_filter(sdf, sigma=smooth_gradient) for sdf in sdf_vol_normal])

    if norm_vectors:
        sdf_vol_normal = sdf_vol_normal / (np.linalg.norm(sdf_vol_normal, axis=0)[None,:]+eps)

    return sdf_vol_normal, sdf_vol

def relabel_slices(labelled, bg_label=0):

    import numpy as np
    max_ID = 0
    labelled_ = []
    for lab in labelled:
        lab[lab>bg_label] = lab[lab>bg_label] + max_ID # only update the foreground!
        labelled_.append(lab)
        max_ID = np.max(lab)+1

    print(max_ID)

    labelled_ = np.array(labelled_)

    return labelled_

def get_cellpose_flow_2D_stack(stack):

    import cellpose
    import numpy as np

    # label_flows, label_flows_dist = cellpose.dynamics.masks_to_flows_cpu(st)
    stack_flow = np.array([cellpose.dynamics.masks_to_flows_cpu(st)[0] for st in stack])

    return stack_flow

def read_pickle(filename):

    import pickle

    with open(filename, 'rb') as output:
        return pickle.load(output)

def write_pickle(savepicklefile, savedict):

    import pickle
    # savepicklefile = os.path.join(savefolder, basename+'_cellpose_combined2D_3D_gradients_probmask.pickle')
    with open(savepicklefile, 'wb') as handle:
        pickle.dump(savedict,
                    handle)
    return []

def parse_tag(string, tag):

    import re

    search = '<'+tag+'.*?>(.+?)</'+tag+'>'
    print(search)

    return re.findall(search, string)

def _ma_average(sig, winsize=3, mode='reflect', avg_func=np.nanmean):

    sig_ = np.pad(sig, [winsize//2, winsize//2], mode=mode)
    sig_out = []
    for ii in np.arange(len(sig)):
        data = sig_[ii:ii+winsize]
        sig_out.append(avg_func(data))
    return np.hstack(sig_out)

def apply_cellpose_model_2D_prob_slice(im_slice,
                                       model,
                                        model_channels,
                                        best_diam=None,
                                        model_invert=False,
                                        test_slice=None,
                                        diam_range=np.arange(15,51,5),
                                        ksize=25,
                                        smoothwinsize=5,
                                        hist_norm=True,
                                        kernel_size=(256,256),
                                        clip_limit=0.01,
                                        fraction_threshold=0.1,
                                        n_proc=48,
                                        bg_remove=False,
                                        use_edge=False):

    # from .filters import var_filter
    # from .gpu import cuda_equalize_adapthist
    import gradient_watershed.filters as grad_filters
    # from grad_filters import var_filter

    # test_slice = im_slice.copy()

    if best_diam is None:

        diam_score = []

        for diam in diam_range[:]:

            img = im_slice.copy()

            _, flow, style = model.cp.eval([img],
                                        channels=model_channels,
                                        batch_size=32,
                                        do_3D=False,
                                        flow_threshold=0.6,
                                        diameter=diam, # this is ok
                                        invert=model_invert) # try inverting?

            # score the content! e.g. sobel, var,
            # prob_score = np.nanmean(var_filter(flow[0][1][0], ksize=ksize)+var_filter(flow[0][1][1], ksize=ksize))
            prob = flow[0][2]
            prob = 1./(1+np.exp(-prob))
            # prob_score = np.nanmean(1./(var_filter(prob, ksize=ksize) + 0.1)*prob)
            # prob_score = np.nanmean(var_filter(prob, ksize=ksize))
            # prob_score = np.median(prob) / np.std(prob)
            # prob_score = np.nanmean(entropy(prob, skmorph.disk(ksize)))
            # prob_score = np.mean(prob) / (np.std(prob)) # signal to noise ratio.
            # prob_score = np.nanmean(entropy(flow[0][1][0]/5., skmorph.disk(ksize))) + np.nanmean(entropy(flow[0][1][1]/5., skmorph.disk(ksize)))

            prob_score = np.nanmean(grad_filters.var_filter(flow[0][1][0], ksize=ksize) + grad_filters.var_filter(flow[0][1][1], ksize=ksize)) # doesn't work weel
            # prob_score = ssim(img, prob) # will this  better select?
            # prob_score = np.abs(pearsonos.path.join(analysisRoot,'masks'r(ndimage.uniform_filter(img, size=ksize).ravel(), prob.ravel())[0])
            # prob_score = np.nanmedian(var_filter(flow[0][1][0], ksize=ksize))+np.nanmedian(var_filter(flow[0][1][1], ksize=ksize))

            # prob_score = mask[0].max()
            # prob_score = np.nanmean(flow[0][2])
            diam_score.append(prob_score)

            # plt.figure(figsize=(5,5))
            # plt.subplot(311)
            # plt.title(str(diam)+ ' '+str(prob_score))
            # plt.imshow(img[:1024,:1024])
            # plt.subplot(312)
            # plt.title(np.mean(flow[0][2]))
            # plt.imshow(flow[0][2][:1024,:1024])
            # plt.subplot(313)
            # plt.imshow(var_filter(flow[0][2], ksize=ksize)[:1024,:1024])
            # plt.show()

        diam_score = np.hstack(diam_score)

        # smooth this.
        diam_score = _ma_average(diam_score, winsize=smoothwinsize)

        # =============================================================================
        #     Compute the best. diameter for this view.
        # =============================================================================
        best_diam = diam_range[np.argmax(diam_score)]

        # plt.figure()
        # plt.plot(diam_range, diam_score, 'o-')
        # plt.show()

    else:
        diam_score = []

    # print('auto determine cell diameter: ', best_diam)

    _, flow, style = model.cp.eval([img],
                                    batch_size=32,
                                    channels=model_channels,
                                    diameter=best_diam,
                                    invert=model_invert,
                                    compute_masks=False)

    # return flow[0][2], flow[0][1], style[0]
    (all_probs, all_flows, all_styles) = (flow[0][2], flow[0][1], style[0])

    return (diam_range, diam_score, best_diam), (all_probs, all_flows, all_styles)

if __name__=="__main__":
    import os
    import numpy as np
    import pylab as plt
    import skimage.io as skio
    from skimage.io import imread, imshow
    import skimage.transform as sktransform
    import skimage.segmentation as sksegmentation
    from skimage.segmentation import clear_border
    from cellpose import models
    import cellpose
    import scipy.ndimage as ndimage
    import skimage.filters as skfilters
    import gradient_watershed.filters as grad_filters
    import gradient_watershed.watershed as grad_watershed
    import gradient_watershed.flows as grad_flows
    import gradient_watershed.gpu as gradient_gpu
    import skimage.restoration as skrestoration
    import stackview
    
    infolders = [f for f in os.listdir(rootfolder) if os.path.isdir(os.path.join(rootfolder, f))]
    for folder in range(len(infolders)):
        subfolder = os.path.join(rootfolder, infolders[folder])
  
        """
        Determine the input files. 
        """
        list = os.listdir(subfolder)

        TIFFall = [file for file in list if 'TIF' and not 'nd' in file]
        rawTIFF = [file for file in TIFFall if not 'thumb' in file]
        TIFF488 = [file for file in rawTIFF if 'TIF' and '488' in file]
        TIFF488.sort()
        TIFF561 = [file for file in rawTIFF if 'TIF' and '561' in file]
        TIFF561.sort()
        #TIFF642 = [file for file in rawTIFF if 'TIF' and '642' in file]
        #TIFF642.sort()

        
        for file in range(len(TIFF561)): 
            # """
            # Readand merge  the image
            # """
            im488 = imread(os.path.join(subfolder, TIFF488[file]))
            im561 = imread(os.path.join(subfolder, TIFF561[file]))
            im = np.dstack((im488,im561))
            
            infile = os.path.join(subfolder, TIFF561[file])
            
            # False is fine.
            hist_norm = False

            basename = os.path.split(infile)[-1].split('_w2561-WF.TIF')[0]
            print('analysing ', basename)
        
            def dask_cuda_rescale(img, zoom, order=1, mode='reflect', chunksize=(512,512,512)):
                import dask.array as da
                im_chunk = da.from_array(img, chunks=chunksize) # make into chunk -> we can then map operation?
                g = im_chunk.map_blocks(gradient_gpu.cuda_rescale, zoom=zoom, order=order, mode=mode)
                # g = im_chunk.map_blocks(ndimage.zoom, zoom=zoom, order=order, mode=mode)
                result = g.compute()
                return result
        
            # this now seems to work...
            def dask_cuda_bg(img, bg_ds=8, bg_sigma=5, chunksize=(512,512,512)):
                import dask.array as da
                # import cucim.skimage.transform as cu_transform # seems to not need this and this seems experimental / temperamental.
        
                im_chunk = da.from_array(img, chunks=chunksize) # make into chunk -> we can then map operation?
                # g = im_chunk.map_blocks(gradient_gpu.bg_normalize, bg_ds=bg_ds, bg_sigma=bg_sigma, dtype=np.float32)  #### so we can't map like this....
                g = im_chunk.map_blocks(gradient_gpu.cuda_rescale, zoom=[1./bg_ds]*len(img.shape), order=1, mode='reflect', dtype=np.float32)
                # g = im_chunk.map_blocks(ndimage.zoom, zoom=[1./bg_ds]*len(img.shape), order=1, mode='reflect', dtype=np.float32)
                # now do the gaussian filter
                g = g.map_overlap(ndimage.gaussian_filter, sigma=bg_sigma, depth=2*bg_sigma, boundary='reflect', dtype=np.float32).compute()  # we need to compute in order to get the scaling.
        
                # we might be able to do the proper scaling in this space....
                im_chunk = da.from_array(g, chunks=chunksize)
                # g = cu_transform.resize(g, np.array(im.shape), preserve_range=True)
                # print(np.hstack(img.shape)/(np.hstack(im_chunk.shape)))
                # g = im_chunk.map_blocks(gradient_gpu.cuda_rescale, zoom=np.hstack(img.shape)/(np.hstack(im_chunk.shape)), order=1, mode='reflect', dtype=np.float32)
                g = im_chunk.map_blocks(gradient_gpu.cuda_rescale, zoom=np.hstack(img.shape)/(np.hstack(im_chunk.shape)), order=1, mode='reflect', dtype=np.float32)
                # g = im_chunk.map_blocks(ndimage.zoom, zoom=np.hstack(img.shape)/(np.hstack(im_chunk.shape)), order=1, mode='reflect', dtype=np.float32)
                result = g.compute()
                result = gradient_gpu.num_bg_correct(img,result, eps=1e-8)
                return result
            """           # threshold_value = skfilters.threshold_otsu(EDU[~np.isnan(EDU)])
                    # threshold_method = "otsu" 

            preprocessing.
            """
            # im_norm = np.max(np.array([grad_filters.normalize(im_ch, pmin=0, pmax=100, clip=True) for im_ch in im]), axis=0)
            im_norm = np.array([grad_filters.normalize(im[...,ch_VCL], clip=True) for ch_VCL in np.arange(im.shape[-1])])
            # im_norm = np.array([grad_filters.normalize(im_ch, clip=True) for im_ch in im])[0]
            im_norm = np.array([ndimage.zoom(imm, zoom=[factor, factor], order=1, mode='reflect') for imm in im_norm])
            
            
            im_norm_2 = np.array([np.mean(im_norm)/(ndimage.gaussian_filter(im_ch, sigma=im_ch.shape[1]//16) + 1)*im_ch for im_ch in im_norm]) # correct background. 
        
            im_norm_2 = grad_filters.normalize(im_norm_2, clip=True) #### operate on this with cellpose
            
        
            comb_vol = im_norm_2.copy()
            # #### recombine into an RGB image.
            # comb_vol = np.concatenate([im_norm_2[None,...],
            #                             im_norm_2[None,...],
            #                             np.zeros_like(im_norm_2)[None,...]], axis=0).transpose(1,2,3,0) #
        
        
            # """
            # add deconvolution.
            # """
            psf_size = 15
            psf = np.zeros((psf_size, psf_size)) 
            psf[psf_size//2, psf_size//2] = 1
            psf = ndimage.gaussian_filter(psf, sigma=1)**2.
            psf = psf/float(np.sum(psf))
            
            # img = convolve2d(img, psf, 'same')
            # rng = np.random.default_rng()
            # img += 0.1 * img.std() * rng.standard_normal(img.shape)
            # deconvolved_img = restoration.unsupervised_wiener(img, psf)
        
            """
            visualize.
            """
            plt.figure(figsize=(15,15))
            plt.imshow(comb_vol[ch_VCL])
            plt.show()
        # =============================================================================
        # =============================================================================
        # #     Auto determine cellpose diameter
        # =============================================================================
        # =============================================================================
            # modelname = 'cyto'
            # modelname = 'cyto2' 
            modelname = 'cyto3'
            # modelname = 'nuclei'
        
            # model_type='cyto' or 'nuclei' or 'cyto2'
            model = models.Cellpose(model_type=modelname, gpu=True) ### set this off for cpu parallel process.
            # # model = models.Cellpose(model_type='cyto2') # this works best for plants?
            # # model = models.Cellpose(model_type='cyto')
            # # model = models.Cellpose(model_type='nuclei')
            # # model = models.Cellpose(model_type='tissuenet')
            # # model = models.Cellpose(model_type='livecell')
            channels = [0,0] # IF YOU HAVE GRAYSCALE
            #channels = [ch_Factin,ch_VCL] # IF YOU HAVE G=cytoplasm and R=nucleus
          
            savefolder = os.path.join(analysisRoot, 'analysis', condition, basename, modelname)
            mkdir(savefolder)
        
            """
            we should save this combined volume....
            """
            skio.imsave(os.path.join(savefolder,
                                      basename+'_cellpose_input.tif'),
                        np.uint8(255*comb_vol)) # this would allow us to test using guided filter.
            angles = ['xy']
        # # =============================================================================
        # # =============================================================================
        # # #     Auto apply cellpose
        # # =============================================================================
        # # =============================================================================
        
            prob_mask = True
            
            im_slice = comb_vol.copy() #.transpose(1,2,0)
                
            im_slice = grad_filters.normalize(im_slice, clip=True)
            # im_slice_edge = skfilters.frangi(im_slice, sigmas=[3], beta=1, gamma=1e-2, black_ridges=False)
            
            im_slice = np.array([skrestoration.unsupervised_wiener(imm, psf)[0] for imm in im_slice]) # this seems better 
            # im_slice = skrestoration.wiener(im_slice, psf, balance=0.1) #[0]
            im_slice = np.clip(im_slice, 0, 1)
            im_slice = im_slice.transpose(1,2,0)
              
            # im_slice = .5*grad_filters.normalize(im_slice, clip=True) + .5*grad_filters.normalize(im_slice_edge, clip=True)
            
            params, (all_probs, all_flows, all_styles) = apply_cellpose_model_2D_prob_slice(im_slice[:,:,ch_VCL],
                                                                                            model,
                                                                                            model_channels=channels,
                                                                                            best_diam=None,
                                                                                            model_invert=False,
                                                                                            test_slice=None,
                                                                                            diam_range=np.arange(10,101,2.5),
                                                                                            ksize=5,
                                                                                            smoothwinsize=5,
                                                                                            hist_norm=False,
                                                                                            kernel_size=(256,256),
                                                                                            clip_limit=0.01,
                                                                                            fraction_threshold=0.1,
                                                                                            n_proc=48,
                                                                                            bg_remove=False,
                                                                                            use_edge=False)
            
        
            plt.figure()
            plt.plot(params[0], params[1])
            plt.show()
        
            ##### the main 
            all_probs = all_probs[0]
            all_flows = all_flows[:,0]
            all_styles = all_styles
            
            all_probs = np.clip(all_probs, -88.72, 88.72)
        
            if prob_mask:
                all_probs = (1./(1.+np.exp(-all_probs)))
            
            all_flows = np.array([ndimage.gaussian_filter(all_flows[ch_VCL], sigma=1) for ch_VCL in np.arange(len(all_flows))])
            all_flows = all_flows/(np.linalg.norm(all_flows, axis=0)[None,...] + 1e-20)
            
            # parse this
            binary = all_probs >= np.maximum(int(skfilters.threshold_multiotsu(all_probs)[0]*10)/10., 0.1) 
            cell_seg_connected_original, cell_seg_connected, tracks, votes_grid_acc = grad_watershed.gradient_watershed2D_binary(binary, 
                                                                                                                                  gradient_img=all_flows.transpose(1,2,0),
                                                                                                                                  momenta=0.98,
                                                                                                                                  n_iter=100)
                                                                                        
            """
            remove bad flow masks
            """
            cell_seg_connected_original, _, _ = grad_flows.remove_bad_flow_masks_2D(cell_seg_connected_original, 
                                                                                      flow=all_flows, 
                                                                                      flow_threshold=1.2,
                                                                                      dtform_method='cellpose_improve',  
                                                                                      fixed_point_percentile=0.01, 
                                                                                      n_processes=4,
                                                                                      power_dist=None,
                                                                                      alpha=0.5, 
                                                                                      filter_scale = 1)
            
            # cell_seg_connected_original = cell_seg_connected_original_new
            """
            remove too small areas. 
            """
            cell_seg_connected_original = cellpose.utils.fill_holes_and_remove_small_masks(cell_seg_connected_original, 
                                                                                            min_size=10) ### this should be implemented to not depend on cellpose!. 
            # cell_seg_connected_original = grad_filters.largest_component_vol_labels_fast(cell_seg_connected_original, connectivity=1)
            
            
            """
            remove masks touching borders
            """    
            #cell_seg_connected_original = clear_border(cell_seg_connected_original)
            
            """
            rescale and mark segementd boundaries to (512,512)
            """
            im_512 = sktransform.resize(im_slice, (512,512),order=0)
            cell_seg_connected_original512 = sktransform.resize(cell_seg_connected_original,(512,512),order=0)
            masks = sksegmentation.mark_boundaries(im_512[:,:,ch_VCL], 
                                                    cell_seg_connected_original512, mode='outer')
            plt.figure()
            plt.imshow(masks)                                         
            plt.show()  
                    
            """
            save the mask
            """
            skio.imsave(os.path.join(savefolder,basename+
                                     '_cyto_mask_overlay.tif'), 
                        np.uint8(255*masks))
            skio.imsave(os.path.join(savefolder,basename+
                                     '_segmentation.tif'), 
                        np.uint16(cell_seg_connected_original))
            
            labelfig, axs = plt.subplots(1,1,figsize=(15,15))
            stackview.imshow(cell_seg_connected_original,labels=True)
            labelfig.savefig(os.path.join(savefolder, basename+ '_labels.png'))
            
        
            # """
            # Overlay each independent nuclear mask over Edu and quantify Edu intensity per cell
            # """
            # # im_scale = ndimage.zoom(im, factor)            
            # # stats = cle.statistics_of_labelled_pixels(im_scale[ch_edu, :,:],cell_seg_connected_original)
            # stats = cle.statistics_of_labelled_pixels(im_norm[ch_Factin, :,:],cell_seg_connected_original)
            # intensity_mean = stats["mean_intensity"]
            
  
            