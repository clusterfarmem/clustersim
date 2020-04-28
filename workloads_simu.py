import numpy as np

class Workload:
    ''' This class is not meant to be used by itself. It's only purpose
        is to provide definitions that are common to all of its children.
    '''
    # These variables are defined in child classes
    # that inherit from this class. Their definition here is
    # just done for clarity.
    wname = None
    ideal_mem = None
    min_ratio = None
    cpu_req = None

    def __init__(self, idd, percent=0, ratio=0, exp_finish=0):

        self.idd = idd  # a unique uint id for this workload
        self.percent = percent # percent of the work done
        self.ratio = ratio
        self.exp_finish = exp_finish # exp finish time (will remove)
        self.prev_ratio = 0 # keep track of previous ratio (id havn't changed, exp_finsh would be the same)
        self.get_gradient()

    def update_percent(self, el_time):
        self.percent = min(self.percent + el_time/self.profile(self.ratio),1)
        assert self.percent < 1

    def update_ratio(self, new_ratio):
        self.ratio = new_ratio

    def update_exp_finish(self, cur_time):
        self.exp_finish = cur_time + (1-self.percent)*self.profile(self.ratio)

    def update(self, L, sid, cur_time, last_time, new_idd=None, new_ratio=1): # ratio = 0 is no remote memory mode
        if last_time == 0:
            el_time = 0
        else:
            el_time = cur_time - last_time
        assert el_time >= 0, '{},{}'.format(cur_time, last_time)

        if (new_idd is not None) and self.idd == new_idd:
            assert self.percent == 0
        else:
            self.update_percent(el_time)
        self.update_ratio(new_ratio)

        if new_ratio != self.prev_ratio:
            if self.exp_finish > 0:
                L.delete_event(self.exp_finish)
            self.update_exp_finish(cur_time)
            self.exp_finish = L.add_event(self.exp_finish, sid, self.wname, self.idd, False)
            self.prev_ratio = new_ratio

    def set_min_ratio(self, new_min_ratio):
        self.min_ratio = new_min_ratio
        self.min_mem = self.min_ratio * self.ideal_mem

    def get_name(self):
        return self.wname + str(self.idd)

    def compute_ratio_from_coeff(self, coeffs, ratio):
        p = 0
        order = len(coeffs)
        for i in range(order):
            p += coeffs[i] * ratio**(order-1-i)
        return p

    def compute_linear_coeffs(self):
        assert len(self.x) == len(self.y)
        self.a = []
        self.b = []
        for i in range(len(self.y)-1):
            tmp_a =  (self.y[i+1] - self.y[i])/(self.x[i+1] - self.x[i])
            tmp_b =  self.y[i] - tmp_a*self.x[i]
            self.a.append(round(tmp_a,2))
            self.b.append(round(tmp_b,2))

    def profile(self,ratio):
        return self.compute_ratio_from_coeff(self.coeff, ratio)*1000 # from second to millisecond

    def get_gradient(self):
        tmp_coeff = self.coeff + [0]
        self.gd_coeff = np.polyder(self.coeff)
        self.mem_gd_coeff = np.polyder(tmp_coeff)

    def gradient(self, ratio):
        return self.compute_ratio_from_coeff(self.gd_coeff, ratio)

    def mem_gradient(self,ratio):
        return self.compute_ratio_from_coeff(self.mem_gd_coeff, ratio)

class Quicksort(Workload):
    wname = "quicksort"
    ideal_mem = 33000
    min_ratio = 0.65
    min_mem = int(min_ratio * ideal_mem)
    cpu_req = 4
    x = [1,0.9,0.8,0.7,0.6,0.5,0.4]
    y = [277.16, 303.402, 310.482, 315.227, 311.71, 326.843, 338.10]
    coeff = [-1984.129, 4548.033, -3588.554, 1048.644, 252.997]

class Linpack(Workload):
    wname = "linpack"
    ideal_mem = 6000
    min_ratio = 0.9
    min_mem = int(min_ratio * ideal_mem)
    cpu_req = 7
    x = [1,0.9,0.8,0.7,0.6,0.5]
    y = [24.174, 25.981, 27.894, 30.681, 33.057, 35.478]
    coeff = [38.52, -77.88, 26.86, 36.70]

class Tfinception(Workload):
    wname = "tf-inception"
    ideal_mem = 6500
    min_ratio = 0.9
    min_mem = int(min_ratio * ideal_mem)
    cpu_req = 6
    x = [1,0.9,0.8,0.7,0.6,0.5]
    y = [405.112, 419.762, 428.091, 436.107, 440.242,475.133]
    coeff = [-1617.416, 3789.953, -2993.734, 1225.477]

class Kmeans(Workload):
    wname = "kmeans"
    ideal_mem = 8000
    min_ratio = 0.75
    min_mem = int(min_ratio * ideal_mem)
    cpu_req = 4
    x = [1,0.9,0.8,0.7,0.6,0.5,0.4]
    y = [359.44, 359.77, 377.9275, 430.557, 508.24, 587.98, 920.575]
    coeff = [-8258.542,  25767.366, -28409.394,  12549.084,  -1289.025]

class Spark(Workload):
    wname = "spark"
    ideal_mem = 10000
    min_ratio = 0.75
    min_mem = int(min_ratio * ideal_mem)
    cpu_req = 7
    x = [1,0.9,0.8,0.7,0.6,0.5]
    y = [74.472, 81.223, 144.567, 222.010, 234.156, 249.249]
    coeff = [4689.05, -10841.59, 7709.92, -1486.13]

class Memaslap(Workload):
    wname = "memaslap"
    ideal_mem = 36000#12288
    min_ratio = 0.5
    min_mem = int(min_ratio * ideal_mem)
    cpu_req = 6
    x = [1,0.9,0.8,0.7,0.6,0.5,0.4]
    y = [908.03, 926.2, 950.87, 1007.09, 1119.16, 1272.82, 1416.51]
    coeff = [-11626.894, 32733.914, -31797.375, 11484.578, 113.33]

def get_workload_class(wname):
    return {'quicksort': Quicksort,
            'linpack': Linpack,
            'tf-inception': Tfinception,
            'spark': Spark,
            'kmeans': Kmeans,
            'memaslap': Memaslap}[wname]
