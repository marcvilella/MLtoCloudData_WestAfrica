import numpy as np
import pandas as pd 
from scipy import signal           
from skimage.measure import compare_ssim
from collections import deque
import cv2
import math, cmath

class Evaluation(object):

    def __init__(self, max_level, windows, levels, threshold):
        self.max_level = max_level # Max_level for PSNR
        self.windows = windows # window size for FSS
        self.levels = levels # threshold levels for FSS
        self.threshold = threshold # threshold for FSS
        self.clear()

    def __call__(self, pred, gt):
        assert (pred.shape == gt.shape)
        
        for i in range(pred.shape[0]):
        
            for im in range(pred.shape[1]):
            
                prediction = self.to2DMatrix(pred[i])
                target = self.to2DMatrix(gt[i])
            
                # PSNR
                #self.psnr += self.compute_psnr(pred[i], gt[i])
                self.psnr += self.compute_psnr(prediction, target)
                
                # SSIM          
                (score, diff) = compare_ssim(prediction, target, full=True)
                #diff = (diff * 255).astype("uint8")
                self.ssim += score
                
                # FSS
                for win_idx in range(len(self.windows)):
                    tmp = self.compute_fss(prediction, target, self.threshold, self.windows[win_idx])
                    #if np.isnan(tmp[2]): self.fss[win_idx] = np.nan 
                    #else: self.fss[win_idx] += tmp[2]
                    if not np.isnan(tmp[2]): self.fss[win_idx] += tmp[2]
                
                # Average Cloud Size
                tmp = self.compute_cloud_areas(prediction)
                self.clouds.extend(tmp)
                tmp = self.compute_cloud_areas(target)
                self.clouds_tar.extend(tmp)
            
                self.count += 1

    def to2DMatrix(self, image):
        # Already one channel
        if image.shape[0] == 1:
            return image[0]
        # Convert to Grayscale
        if image.shape[0] == 3:
            return (image[0] +  image[1] +  image[2]) / 3
        return image

    def Loss(self):
        return self.loss / max(1, self.count)

    def PSNR(self):
        return self.psnr / max(1, self.count)
        
    def SSIM(self):
        return self.ssim / max(1, self.count)
        
    def FSS(self):
        return [fss / max(1, self.count) for fss in self.fss]
        
    def CloudSize(self, mode = 2):
        if len(self.clouds) == 0: return 0
        
        if mode == 0:
            return self.clouds
        elif mode == 1:
            return np.mean(self.clouds)
        elif mode == 2:
            return np.median(self.clouds)
            
    def CloudSizeTarget(self, mode = 2):        
        if len(self.clouds_tar) == 0: return 0
        
        if mode == 0:
            return self.clouds_tar
        elif mode == 1:
            return np.mean(self.clouds_tar)
        elif mode == 2:
            return np.median(self.clouds_tar)
        
    def MovingDistance(self, mode, method = 0, width = 256, height = 256):
        if min(len(self.dA[method]), len(self.dA_upper[method]), len(self.dA_middle[method]), len(self.dA_lower[method])) == 0:
            return [[0,0], [0,0], [0,0], [0,0]]
    
        # Original vectors
        if mode is 0:
            return self.dA[method], self.dA_upper[method], self.dA_middle[method], self.dA_lower[method]
        # Vector averages
        if mode is 1:
            return (
                np.sum(self.dA[method], axis=0) / len(self.dA[method]), 
                np.sum(self.dA_upper[method], axis=0) / len(self.dA_upper[method]), 
                np.sum(self.dA_middle[method], axis=0) / len(self.dA_middle[method]), 
                np.sum(self.dA_lower[method], axis=0) / len(self.dA_lower[method]))
        
        # Speed and direction
        if mode is 2:
            tmp_dA = np.sum(self.dA[method], axis=0) / len(self.dA[method])
            tmp_dA = complex(tmp_dA[0], tmp_dA[1])
            tmp_dA_upp = np.sum(self.dA_upper[method], axis=0) / len(self.dA_upper[method])
            tmp_dA_upp = complex(tmp_dA_upp[0], tmp_dA_upp[1])
            tmp_dA_mid = np.sum(self.dA_middle[method], axis=0) / len(self.dA_middle[method]) 
            tmp_dA_mid = complex(tmp_dA_mid[0], tmp_dA_mid[1])
            tmp_dA_low = np.sum(self.dA_lower[method], axis=0) / len(self.dA_lower[method])
            tmp_dA_low = complex(tmp_dA_low[0], tmp_dA_low[1])
            return (                
                [abs(tmp_dA), math.degrees(cmath.phase(tmp_dA))],
                [abs(tmp_dA_upp), math.degrees(cmath.phase(tmp_dA_upp))],
                [abs(tmp_dA_mid), math.degrees(cmath.phase(tmp_dA_mid))],
                [abs(tmp_dA_low), math.degrees(cmath.phase(tmp_dA_low))],
            )
            
        # Speed and direction in km
        if mode is 3:
            conv_width_p2km = 4445 / width
            conv_height_p2km = 3334 / height
            tmp_dA = np.sum(self.dA[method], axis=0) / len(self.dA[method])
            tmp_dA = complex(tmp_dA[0]*conv_width_p2km, tmp_dA[1]*conv_height_p2km)
            tmp_dA_upp = np.sum(self.dA_upper[method], axis=0) / len(self.dA_upper[method])
            tmp_dA_upp = complex(tmp_dA_upp[0]*conv_width_p2km, tmp_dA_upp[1]*conv_height_p2km)
            tmp_dA_mid = np.sum(self.dA_middle[method], axis=0) / len(self.dA_middle[method]) 
            tmp_dA_mid = complex(tmp_dA_mid[0]*conv_width_p2km, tmp_dA_mid[1]*conv_height_p2km)
            tmp_dA_low = np.sum(self.dA_lower[method], axis=0) / len(self.dA_lower[method])
            tmp_dA_low = complex(tmp_dA_low[0]*conv_width_p2km, tmp_dA_low[1]*conv_height_p2km)
            return (                
                [abs(tmp_dA), math.degrees(cmath.phase(tmp_dA))],
                [abs(tmp_dA_upp), math.degrees(cmath.phase(tmp_dA_upp))],
                [abs(tmp_dA_mid), math.degrees(cmath.phase(tmp_dA_mid))],
                [abs(tmp_dA_low), math.degrees(cmath.phase(tmp_dA_low))],
            )
        
        
    def Results(self):
        return self.PSNR(), self.SSIM(), self.FSS(), self.CloudSize()

    def clear(self):
        self.loss = 0
        self.psnr = 0
        self.ssim = 0
        self.fss = [0 for i in range(len(self.windows))]
        self.clouds = []
        self.clouds_tar = []
        self.dA, self.dA_upper, self.dA_middle, self.dA_lower = [[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]]
        self.count = 0
    
    def addLoss(self, loss):
        self.loss += loss
    
    def compute_psnr(self, pred, targ):
        # Ensure that pixes are between 0 and 1  for both, prediction and target (equivalent to the 0-255 for images)        
        if np.amax(pred) > 1 or np.amin(pred) < 0:
            f_min, f_max = np.amin(pred), np.amax(pred)
            pred = (pred - f_min) / (f_max - f_min)
        
        if np.amax(targ) > 1 or np.amin(targ) < 0:
            f_min, f_max = np.amin(targ), np.amax(targ)
            targ = (targ - f_min) / (f_max - f_min)
        
        mse = np.mean((np.array(targ, dtype=np.float32) - np.array(pred, dtype=np.float32)) ** 2)
        if mse == 0:
            return 100
            
        max_value = 1.0
        return 20 * np.log10(max_value / (np.sqrt(mse)))
        
    
    def compute_integral_table(self, field): 
        return field.cumsum(1).cumsum(0) 
        
    def ffourier_filter(self, field, n): 
        return signal.fftconvolve(field, np.ones((n, n))) 
    
    def integral_filter(self, field, n, table=None): 
        """
        Fast summed area table version of the sliding accumulator. 
        :param field: nd-array of binary hits/misses. 
        :param n: window size. 
        """
    
        w = n // 2 
        
        if w < 1.: 
            return field 
        
        if table is None: 
            table = self.compute_integral_table(field) 
        
        r, c = np.mgrid[0:field.shape[0], 0:field.shape[1]] 
        
        r = r.astype(np.int) 
        c = c.astype(np.int) 
        w = np.int(w) 
        
        r0, c0 = (np.clip(r - w, 0, field.shape[0] - 1), np.clip(c - w, 0, field.shape[1] - 1)) 
        r1, c1 = (np.clip(r + w, 0, field.shape[0] - 1), np.clip(c + w, 0, field.shape[1] - 1)) 
        
        integral_table = np.zeros(field.shape).astype(np.int64)
        integral_table += np.take(table, np.ravel_multi_index((r1, c1), field.shape)) 
        integral_table += np.take(table, np.ravel_multi_index((r0, c0), field.shape)) 
        integral_table -= np.take(table, np.ravel_multi_index((r0, c1), field.shape)) 
        integral_table -= np.take(table, np.ravel_multi_index((r1, c0), field.shape)) 
        
        return integral_table 
            
    def fourier_fss(self, fcst, obs, threshold, window): 
        """
        Compute the fraction skill score using convolution. 
        :paramfcst: nd-array, forecast field. 
        :paramobs: nd-array, observation field. 
        :param window: integer, window size. 
        :return: tuple of FSS numerator, denominator and score. 
        """
        
        fhat = self.fourier_filter(fcst > threshold, window) 
        ohat = self.fourier_filter(obs > threshold, window) 
        num = np.nanmean(np.power(fhat - ohat, 2))
        denom = np.nanmean(np.power(fhat, 2) + np.power(ohat, 2)) 
        
        return num, denom, 1.-num/denom 
        
    def compute_fss(self, fcst, obs, threshold, window, fcst_cache=None, obs_cache=None): 
        """
        Compute the fraction skill score using summed area tables
        :paramfcst: nd-array, forecast field. 
        :paramobs: nd-array, observation field. 
        :param window: integer, window size. 
        :return: tuple of FSS numerator, denominator and score. 
        """
        
        fhat = self.integral_filter(fcst > threshold, window, fcst_cache) 
        ohat = self.integral_filter(obs > threshold, window, obs_cache) 
        num = np.nanmean(np.power(fhat - ohat, 2)) 
        denom = np.nanmean(np.power(fhat, 2) + np.power(ohat, 2)) 
        
        return num, denom, 1.-num/denom 
            
    def fss_frame(self, fcst, obs, windows, levels): 
        """
        Compute the fraction skill score data-frame. 
        :paramfcst: nd-array, forecast field. 
        :paramobs: nd-array, observation field. 
        :param window: list, window sizes. 
        :param levels: list, threshold levels. 
        :return: list, dataframes of the FSS: numerator,denominator and score. 
        """
        
        num_data, den_data, fss_data = [], [], [] 
        
        for level in self.levels:
            ftable = self.compute_integral_table(fcst > level)
            otable = self.compute_integral_table(obs > level) 
            
            _data = [compute_fss(fcst, obs, level, w, ftable, otable) for w in self.windows] 
            
            num_data.append([x[0] for x in _data]) 
            den_data.append([x[1] for x in _data]) 
            fss_data.append([x[2] for x in _data]) 
            
        return (pd.DataFrame(num_data, index=levels, columns=windows), 
                    pd.DataFrame(den_data, index=levels, columns=windows), 
                    pd.DataFrame(fss_data, index=levels, columns=windows)) 

    #
    # CLOUDS
    #
    
    def convert2Image(self, data):
    
        # if not uint8
        if type(data[0,0]) is not np.uint8:
            
            # Normalize
            f_min, f_max = np.amin(data), np.amax(data)
            if f_min == f_max: return data.round().astype(np.uint8)
            data = (data - f_min) / (f_max - f_min)
            
            # Convert to uint8
            data = (data * 255).round().astype(np.uint8)
    
        # Reshape matrix to be as an 'Image' type
        return cv2.resize(data, (data.shape[0], data.shape[1]))
    
    def compute_cloud_areas(self, img):
        # Convert to Image
        img = self.convert2Image(img)
        
        # Find areas
        blur = cv2.GaussianBlur(img, (5,5), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        im2,_ = cv2.findContours(thresh, 1, 2)
        
        return [cv2.contourArea(im) for im in im2][:-1]
    
    def compute_cloud_movement_batch(self, next_batch, prev_batch):
        assert (next_batch.shape == prev_batch.shape)
        
        for i in range(next_batch.shape[0]):
        
            next_img = self.to2DMatrix(next_batch[i])
            previous_img = self.to2DMatrix(prev_batch[i])
            
            self.compute_cloud_movement(next_img, previous_img, 0)
            self.compute_cloud_movement(next_img, previous_img, 1)
            self.compute_cloud_movement(next_img, previous_img, 2)
    
    def compute_cloud_movement(self, next_img, previous_img, mode):
        
        # Convert to Image
        next_img = self.convert2Image(next_img)
        previous_img = self.convert2Image(previous_img)
        imgs = [previous_img, next_img]
        
        # General Variables 
        lowerBoundary, upperBoundary = (0, 0, 15), (0, 0, 255)
        pts, pts_ocean, pts_main, pts_moutains = deque(maxlen=2), deque(maxlen=2), deque(maxlen=2), deque(maxlen=2)
        
        img_width = len(next_img[0])
        img_widht_5 = int(img_width / 8)
        img_height = len(next_img)
        img_height_5 = int(img_height / 6)
        
        # Loop for both images
        for image in imgs:
        
            # Clean Image and Get Masks
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            mask, thresh = self.clean_image(image, lowerBoundary, upperBoundary)
            mask_ocean, mask_mainland, mask_mountain = self.filter_areas(mask, img_height_5, img_widht_5)

            # Contours
            cnt = self.getBiggestCloud(mask)
            cnt_ocean = self.getBiggestCloud(mask_ocean)
            cnt_mainland = self.getBiggestCloud(mask_mainland)
            cnt_mountains = self.getBiggestCloud(mask_mountain)
            
            # Calculate Centers and Compute distances
            if cnt is not None: 
                center = self.getCenter(cnt, image, 12, mode = mode)
                if center is not None: pts.appendleft(center)
                else: pts_ocean.clear()
            else: pts.clear()
            if (len(pts) >= 2):
                dX, dY = self.getMovement(pts)
                if dX is not None:
                    self.dA[mode].append([dX, dY, cv2.contourArea(cnt)])
                else:
                    pts.clear()
            
            if cnt_ocean is not None: 
                center = self.getCenter(cnt_ocean, image, 6, mode = mode)
                if center is not None: pts_ocean.appendleft(center)
                else: pts_ocean.clear()
            else: pts_ocean.clear()
            if (len(pts_ocean) >= 2):
                dX, dY = self.getMovement(pts_ocean)
                if dX is not None:
                    self.dA_upper[mode].append([dX, dY, cv2.contourArea(cnt_ocean)])
                else:
                    pts_ocean.clear()
            
            if cnt_mainland is not None: 
                center = self.getCenter(cnt_mainland, image, 6, mode = mode)
                if center is not None: pts_main.appendleft(center)
                else: pts_main.clear()
            else: pts_main.clear()
            if (len(pts_main) >= 2):
                dX, dY = self.getMovement(pts_main)
                if dX is not None:
                    self.dA_middle[mode].append([dX, dY, cv2.contourArea(cnt_mainland)])
                else:
                    pts_main.clear()
                
            if cnt_mountains is not None: 
                center = self.getCenter(cnt_mountains, image, 6, mode = mode)
                if center is not None: pts_moutains.appendleft(center)
                else: pts_moutains.clear()
            else: pts_moutains.clear()
            if (len(pts_moutains) >= 2):
                dX, dY = self.getMovement(pts_moutains)
                if dX is not None:
                    self.dA_lower[mode].append([dX, dY, cv2.contourArea(cnt_mountains)])
                else:
                    pts_moutains.clear()
    
    
    def clean_image(self, image, lowerBoundary, upperBoundary):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blur = cv2.GaussianBlur(gray, (5,5), 0)

        mask = cv2.inRange(blur, lowerBoundary, upperBoundary)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        return mask, thresh

    def filter_areas(self, mask, img_height_5, img_widht_5):
        # Filter by areas
        img_ocean, img_mainland, img_mountains = mask.copy(), mask.copy(), mask.copy()
        # Filter by Ocean (-20W to -15W and whole height, -20W to 10E and 0 to 5N)
        img_ocean[2*img_height_5:, :] = 0
        # Filter by Mainland
        img_mainland[:2*img_height_5, :] = 0
        img_mainland[4*img_height_5:, :] = 0
        # Filter by Mountains (0 to 20E and 15 to 30N)
        img_mountains[:4*img_height_5, :] = 0
        
        return img_ocean, img_mainland, img_mountains

    def getBiggestCloud(self, mask):
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        if(len(cnts) > 0): return max(cnts, key=cv2.contourArea)
        else: return None
        
    def getCenter(self, contour, image, radius_min = 10, mode = 0):
        # find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        M = cv2.moments(contour)
        
        # only proceed if the radius meets a minimum size
        if radius < radius_min or M["m00"] == 0: return None
        
        # Center
        if mode == 0:
            return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # Most left
        elif mode == 1: 
            return tuple(contour[contour[:,:,0].argmin()][0])
        # Point with highest value
        elif mode == 2:
            mask = np.zeros_like(image)
            cv2.fillPoly(mask, pts =[contour], color=(255,255,255))
            
            points = np.where(mask == 255)
            pixels = [0,0]
            pixel_value = 0
            for i in range(len(points[0])):
                tmp = image[points[0][i], points[1][i]][0]
                if tmp > pixel_value: 
                    pixels = [points[1][i], points[0][i]]
                    pixel_value = tmp
            return tuple(pixels)

    def getMovement(self, pts):
        # if either of the tracked points are None, ignore
        # them
        if pts[0] is None or pts[1] is None:
            return None, None
        # check to see if enough points have been accumulated in
        # the buffer
        if pts[-2] is not None:
            # compute the difference between the x and y
            dX = pts[0][0] - pts[1][0]
            dY = pts[1][1] - pts[0][1]
                
        # Check maximum speed to "ensure" same cloud -> A change of more than 6px would mean speeds of more than 200kmh horitzontally and 150km vertically
        if abs(dX) > 5 or abs(dY) > 5: 
            return None, None
        
        return dX, dY    
