import pprint

import fire
import tabulate
from collections import deque
import math
import dataclasses
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import webbrowser
import numpy as np
from ghe_exp import graph_config
import main_exp_plots


sb__W_m2K4 = 5.67e-8  # W/(m^2K^4)


S_cylinder  = 0.102260
k_black_pla = 0.09


def gen_air_tc_byC():
    """Generate lookup table for air thermal conductivity by air temp"""
    import numpy as np

    # known values from https://www.me.psu.edu/cimbala/me433/Links/Table_A_9_CC_Properties_of_Air.pdf
    known_t = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100])

    known_k = np.array(
        [0.02364, 0.02401, 0.02439, 0.02476, 0.02514, 0.02551, 0.02588, 0.02625, 0.02662, 0.02699, 0.02735, 0.02808,
         0.02881, 0.02953, 0.03024, 0.03095])

    t_range = np.arange(0, 101, 1)

    k_interp = np.interp(t_range, known_t, known_k)

    return list(k_interp)


def get_air_thermal_conductivity(air_temp__C):
    # linearly interpolate between array values to get any amount
    temp_floor = int(air_temp__C)
    temp_ceil = temp_floor + 1
    frac = air_temp__C - temp_floor
    return materials['air']['tc_byC'][temp_floor] + frac * (materials['air']['tc_byC'][temp_ceil] - materials['air']['tc_byC'][temp_floor])


materials = {
    'air': {
        # thermal conductivity (W/mK) by air temperature in C
        'tc_byC': gen_air_tc_byC(),
    },
    'black_pla': {
        'name': 'black_pla',
        # thermal conductivity
        'tc__W_mK': k_black_pla,
        # specific heat capacity
        'shc__J_kgK': 1800,
        # density, https://bitfab.io/blog/3d-printing-materials-densities/
        'd__g_cm3': 1.24,
        'solar_tra': [0, 0.05, 0.95],
        'ir_tra': [0, 0.05, 0.95],
    },
    'borosilicate': {
        'name': 'borosilicate',
        'tc__W_mK': 1.15,  # mid of 1.1-1.2
        # heat capacity of the disk
        'disk_hc__J_k': 7.95,  # mid of [7.0, 8.9]
        'solar_tra': [0.905, 0.020, 0.075],
        # 'solar_tra': [0.909, 0.091, 0.000],  # even with no solar abs, boro surf still remains hottest
        # 'solar_tra': [0.9, 0.0, 0.1],        # counterfactual of all same solar transmission
        'ir_tra':    [0.00,  0.06,  0.94],
    },
    'sapphire': {
        'name': 'sapphire',
        'tc__W_mK': 25.85,  # mid of 18.7-33.0
        'disk_hc__J_k': 12.55,  # mid of [11.7, 13.4],
        'solar_tra': [0.832, 0.140, 0.028],
        # 'solar_tra': [0.9, 0.0, 0.1],        # counterfactual of all same solar transmission
        'ir_tra':    [0.01,  0.30, 0.69],
    },
    'caf2': {
        'name': 'caf2',
        'tc__W_mK': 10.25,  # mid of [9.7, 10.8],
        'disk_hc__J_k': 11.5,  # mid of [10.8, 11.3],
        'solar_tra': [0.914, 0.08, 0.006],
        # 'solar_tra': [0.9, 0.0, 0.1],        # counterfactual of all same solar transmission
        'ir_tra':    [0.37,  0.06, 0.57],
    },
}


@dataclasses.dataclass
class Component:
    # heat capacity
    hc__J_K: float
    # initial temperature
    initial_T__K: float

    # solar trans, reflection, absorption
    solar_tra: list[float]
    # infrared trans, reflection, absorption
    ir_tra: list[float]

    # internal energy
    ie__J: float = 0
    # temperature
    T__K: float = None

    @property
    def T__C(self) -> float:
        return self.T__K - 273.15

    @property
    def solar_trans(self):
        return self.solar_tra[0]

    @property
    def solar_refl(self):
        return self.solar_tra[1]

    @property
    def solar_abs(self):
        return self.solar_tra[2]

    @property
    def ir_trans(self):
        return self.ir_tra[0]

    @property
    def ir_refl(self):
        return self.ir_tra[1]

    @property
    def ir_abs(self):
        return self.ir_tra[2]

    @property
    def ir_emissivity(self):
        return self.ir_abs

    def initialize(self):
        self.ie__J = 0
        self.update_ie(delta_ie__J=0)

    def update_ie(self, delta_ie__J: float):
        self.ie__J += delta_ie__J

        # how much has K changed based on heat capacity?
        deltaK = self.ie__J / self.hc__J_K
        self.T__K = self.initial_T__K + deltaK


class ModelSurfPane:
    def __init__(
            self,
            # start temp
            Tstart__C,
            # environ vars: convection & conduction thru cylinder
            pane_h__W_m2K, cylinder__S_m,
            # pla material
            surf_material,
            # pane material
            pane_material,
            # whether to use 2 layers for pane
            two_layer_pane=False,
    ):
        # initial temp in K
        Tstart__K = Tstart__C + 273.15

        # model surf as a 1mm thick, 4cm *diameter* disk, which
        # loses heat to cylinder below with constant-temp bottom and sides
        self.pane_radius__m = 0.020
        # the thickness can be adjusted to roughly model the heat capacity of the cylinder
        self.surf_thickness__m = 0.001
        black_pla_volume__m3 = math.pi * (self.pane_radius__m**2) * self.surf_thickness__m
        # calc mass from volume & density d__g_cm3
        pla_density__kg_m3 = surf_material['d__g_cm3'] * 1000
        black_pla_mass__kg = pla_density__kg_m3 * black_pla_volume__m3
        black_pla_hc__J_K = black_pla_mass__kg * surf_material['shc__J_kgK']

        print("Black PLA surf heat capacity (J/K):", black_pla_hc__J_K)

        # model pane as isothermal 2mm, 4cm radius disk, 1mm above the surf
        self.pane_thickness__m = 0.002
        self.pane_material = pane_material
        self.air_gap__m = 0.001

        self.pieces: dict[str, Component] = {
            'surf': Component(
                initial_T__K=Tstart__K,
                hc__J_K=black_pla_hc__J_K,
                solar_tra=surf_material['solar_tra'],
                ir_tra=surf_material['ir_tra'],
            ),
        }

        self.two_layer_pane = two_layer_pane
        if self.two_layer_pane:
            self.pieces.update({
                'panetop': Component(
                    initial_T__K=Tstart__K,
                    hc__J_K=pane_material['disk_hc__J_k']/2,  # half the heat capacity
                    solar_tra=pane_material['solar_tra'],
                    ir_tra=pane_material['ir_tra'],
                ),
                'panebot': Component(
                    initial_T__K=Tstart__K,
                    hc__J_K=pane_material['disk_hc__J_k']/2,
                    solar_tra=pane_material['solar_tra'],
                    ir_tra=pane_material['ir_tra'],
                ),
            })
        else:
            self.pieces.update({
                'pane': Component(
                    initial_T__K=Tstart__K,
                    hc__J_K=pane_material['disk_hc__J_k'],
                    solar_tra=pane_material['solar_tra'],
                    ir_tra=pane_material['ir_tra'],
                ),
            })

        for piece in self.pieces.values():
            piece.initialize()
            assert 0 <= piece.solar_abs <= 1

        # time step
        if self.two_layer_pane:
            self.step__s = 1 / 3
        else:
            self.step__s = 1 / 5

        # current time
        self.time__s = 0
        self.n_steps = 0

        # convective coefficient for pane losing heat to the outside air
        self.pane_h__W_m2K = pane_h__W_m2K

        # shape factor for cylinder
        # prefer higher shape factor vs. higher conv loss, seems more sensible
        self.cylinder__S_m = cylinder__S_m

        # view factors from surf to pane, separated by given mm
        # (from https://sterad.net/)
        vf_disks_mms = {
            # 1mm above surf is bottom of pane
            1: 0.9512343774406435,
            # 3mm above surf is top of pane
            3: 0.8608287165990133,
            # 6mm above surf is the hole to the sky
            6: 0.74164377375765,

            # # 3mm above the top of the pane is the sky
            # 3: 0.8608287165990133,
            # 5mm above the bottom of the pane is the sky
            5: 0.7793044453656703,

            # extra for ref
            4: 0.8190024875775822,
            7: 0.7059310411756783,
        }
        self.vfs = {
            'surf->botwall': 1 - vf_disks_mms[1],
            'surf->panewall': vf_disks_mms[1] - vf_disks_mms[3],
            'surf->topwall': vf_disks_mms[3] - vf_disks_mms[6],
            'surf->sky': vf_disks_mms[6],
            'botpane->botwall': 1 - vf_disks_mms[1],
            'botpane->surf': vf_disks_mms[1],
            'toppane->topwall': 1 - vf_disks_mms[3],
            'toppane->sky': vf_disks_mms[3],
        }
        # print(json.dumps(self.vfs,indent=2))
        assert abs(sum([
            self.vfs['surf->botwall'],
            self.vfs['surf->panewall'],
            self.vfs['surf->topwall'],
            self.vfs['surf->sky'],
        ]) - 1) < 0.00001
        assert abs(sum([
            self.vfs['botpane->botwall'],
            self.vfs['botpane->surf'],
        ]) - 1) < 0.00001
        assert abs(sum([
            self.vfs['toppane->topwall'],
            self.vfs['toppane->sky'],
        ]) - 1) < 0.00001

        # logs
        self.latest__W = {}

    @property
    def surf_pane_area__m2(self):
        return math.pi * (self.pane_radius__m**2)

    def Q_conduction_surf_cylinder__W(self, Ta__C):
        # use shape factor to approximate
        # $ Q = k S (T - T_a) $
        Tdiff = self.pieces['surf'].T__C - Ta__C
        Q = materials['black_pla']['tc__W_mK'] * self.cylinder__S_m * Tdiff
        # if surf is hotter it will have positive Q, and be losing heat, so negative to be away from
        return -Q

    def Q_conduction_surf_pane__W(self):
        Tpane__C = self.pieces['panebot' if self.two_layer_pane else 'pane'].T__C

        # Q = k (A/L) (T - T_a)
        # through the air in the air gap
        Tdiff = self.pieces['surf'].T__C - Tpane__C

        # consider gapAirC to be avg of pane and surf
        Tairgap__C = (Tpane__C + self.pieces['surf'].T__C) / 2
        # k_air__W_mK = materials['air']['tc_byC'][int(round(Tairgap__C))]
        k_air__W_mK = get_air_thermal_conductivity(Tairgap__C)

        Q = k_air__W_mK * (self.surf_pane_area__m2 / self.air_gap__m) * Tdiff
        # positive = it's losing heat, negate
        return -Q

    def Qpanebot_cond_top__W(self):
        # conduction from bottom pane to top pane, through 2mm thickness
        if not self.two_layer_pane:
            # lumped isothermal
            return 0

        Tpanebot__C = self.pieces['panebot'].T__C
        Tpanetop__C = self.pieces['panetop'].T__C

        # Q = k (A/L) (T - T_a)
        # through the 2mm thick pane material
        Tdiff = Tpanebot__C - Tpanetop__C
        k_pane__W_mk = self.pane_material['tc__W_mK']

        Q = k_pane__W_mk * (self.surf_pane_area__m2 / self.pane_thickness__m) * Tdiff
        # positive = it's losing heat, negate
        return -Q

    def Q_convection_pane_air__W(self, Ta__C):
        # Q = h A (T - Ta)
        Tpane__C = self.pieces['panetop' if self.two_layer_pane else 'pane'].T__C
        Tdiff = Tpane__C - Ta__C
        Q = self.pane_h__W_m2K * self.surf_pane_area__m2 * Tdiff
        return -Q

    def avg_pane__K(self):
        if self.two_layer_pane:
            return (self.pieces['panetop'].T__K + self.pieces['panebot'].T__K) / 2

        return self.pieces['pane'].T__K

    def avg_pane__C(self):
        if self.two_layer_pane:
            return (self.pieces['panetop'].T__C + self.pieces['panebot'].T__C) / 2

        return self.pieces['pane'].T__C

    def Qir_surf__W(self, sky__W_m2, Ta__C):
        bits = {}

        TsurfC = self.pieces['surf'].T__C
        TsurfK = self.pieces['surf'].T__K

        # bottom wall is same as surf
        TbotwallK = TsurfK
        # pane wall is same as pane
        TpanewallK = self.avg_pane__K()
        # upper wall is same as air
        TtopwallC = Ta__C
        TtopwallK = TtopwallC + 273.15

        # surf has same emissivity for all types of radiation
        e_surf = self.pieces['surf'].ir_emissivity
        # and reflections
        r_surf = self.pieces['surf'].ir_refl
        # no absorption
        assert abs(e_surf + r_surf - 1) < 0.00001 and 0 <= e_surf <= 1 and 0 <= r_surf <= 1

        # surf's own emission
        surf_emission = e_surf * sb__W_m2K4 * (TsurfK ** 4) * self.surf_pane_area__m2

        # surf has four view factors:
        # 1 - bottom walls, emits according to its temp and receives same from bottom walls
        e_walls = e_surf
        botwall_emission = e_walls * sb__W_m2K4 * (TbotwallK ** 4) * self.surf_pane_area__m2
        # botwall emission would be scaled by surf absorption but it should zero out through reflections etc
        # so we leave it like this
        Qsurf_botwall = -surf_emission + botwall_emission
        # should always be 0
        assert abs(Qsurf_botwall) < 0.00001

        # next 3 pieces are all through pane
        TpaneK = self.avg_pane__K()

        # top & bot have same ir props so doesn't matter the key here
        panekey = 'panebot' if self.two_layer_pane else 'pane'
        e_pane = self.pieces[panekey].ir_emissivity
        pane_emission = e_pane * sb__W_m2K4 * (TpaneK ** 4) * self.surf_pane_area__m2

        r_pane = self.pieces[panekey].ir_refl
        t_pane = self.pieces[panekey].ir_trans

        def exchange_through_pane(other_side_emission):
            Qexch = 0
            # exchange through pane is done like this:
            # surf emits its own emission
            Qexch += -surf_emission
            bits = {}
            bits['surf_emission'] = -surf_emission
            # it receives:
            # 1 - the pane's own emission (once)
            Qexch += pane_emission * e_surf
            bits['surf_paneemit_abs'] = pane_emission * e_surf

            # 2 - the 'other side' , *through* the pane (whatever it transmits)
            Qexch += other_side_emission * t_pane * e_surf
            bits['surf_otherside_abs'] = other_side_emission * t_pane * e_surf

            # 3 - one reflection of its own emission from the pane, assume all of this is absorbed
            Qexch += surf_emission * r_pane
            bits['surf_surfrefl_abs'] = surf_emission * r_pane

            return Qexch, bits

        # 2 - pane walls
        panewall_emission = e_walls * sb__W_m2K4 * (TpanewallK ** 4) * self.surf_pane_area__m2
        Qsurf_panewall, panewall_bits = exchange_through_pane(
            other_side_emission=panewall_emission,
        )
        panewall_bits['vf'] = self.vfs['surf->panewall']

        # 3 - top walls
        topwall_emission = e_walls * sb__W_m2K4 * (TtopwallK ** 4) * self.surf_pane_area__m2
        Qsurf_topwall, topwall_bits = exchange_through_pane(
            other_side_emission=topwall_emission,
        )
        topwall_bits['vf'] = self.vfs['surf->topwall']

        # 4 - sky hole
        sky_emission = sky__W_m2 * self.surf_pane_area__m2
        Qsurf_sky, sky_bits = exchange_through_pane(
            other_side_emission=sky_emission,
        )
        sky_bits['vf'] = self.vfs['surf->sky']

        # scale by the view factors and we are good
        Qsurf_all = (
            self.vfs['surf->botwall'] * Qsurf_botwall +
            self.vfs['surf->panewall'] * Qsurf_panewall +
            self.vfs['surf->topwall'] * Qsurf_topwall +
            self.vfs['surf->sky'] * Qsurf_sky
        )
        # def linebit(Q, vf):
        #     return [
        #         float("%.2f" % (Q/self.surf_pane_area__m2)),
        #         float("%.4f" % vf),
        #         float("%.2f" % (Q*vf/self.surf_pane_area__m2)),
        #     ]

        bits.update({
            # '- Qsurf_ir_botwall ': linebit(Qsurf_botwall, self.vfs['surf->botwall']),
            # '- Qsurf_ir_panewall': linebit(Qsurf_panewall, self.vfs['surf->panewall']),
            # '- Qsurf_ir_topwall ': linebit(Qsurf_topwall, self.vfs['surf->topwall']),
            # '- Qsurf_ir_air     ': linebit(Qsurf_sky, self.vfs['surf->sky']),
            '- (Qsurf_ir_emit)': float((sum(bit['surf_emission']*bit['vf'] for bit in [panewall_bits, topwall_bits, sky_bits]))),
            '- (Qsurf_ir_paneabs)': float((sum(bit['surf_paneemit_abs']*bit['vf'] for bit in [panewall_bits, topwall_bits, sky_bits]))),
            '- (Qsurf_ir_othersideabs)': float((sum(bit['surf_otherside_abs']*bit['vf'] for bit in [panewall_bits, topwall_bits, sky_bits]))),
            '- (Qsurf_ir_surfreflabs)': float((sum(bit['surf_surfrefl_abs']*bit['vf'] for bit in [panewall_bits, topwall_bits, sky_bits]))),
            'Qsurf_ir': Qsurf_all,
        })
        return Qsurf_all, bits

    def Qir_pane__W(self, sky__W_m2, Ta__C):
        # similar as for surf but with two view factors: to bottom wall and to surf
        bits = {}

        TsurfK = self.pieces['surf'].T__K

        if self.two_layer_pane:
            TpanebotK = self.pieces['panebot'].T__K
            TpanetopK = self.pieces['panetop'].T__K
            e_pane = self.pieces['panetop'].ir_emissivity  # doesn't matter
        else:
            # bot & top at same temp
            TpanebotK = TpanetopK = self.pieces['pane'].T__K
            e_pane = self.pieces['pane'].ir_emissivity

        panebot_emission = e_pane * sb__W_m2K4 * (TpanebotK ** 4) * self.surf_pane_area__m2
        panetop_emission = e_pane * sb__W_m2K4 * (TpanetopK ** 4) * self.surf_pane_area__m2

        # Bottom side:
        # 1 - towards bottom walls
        e_surf_walls = self.pieces['surf'].ir_emissivity
        r_surf_walls = self.pieces['surf'].ir_refl
        # bottom wall is same temp as surf
        TbotwallK = TsurfK
        botwall_emission = e_surf_walls * sb__W_m2K4 * (TbotwallK ** 4) * self.surf_pane_area__m2
        # we emit out, and receive bot wall emission that we absorb
        Qbotpane_botwall = -panebot_emission + botwall_emission * e_pane
        # also receive one reflection off walls that we absorb fully
        Qbotpane_botwall += panebot_emission * r_surf_walls
        botwall_bits = {
            'botpane_emission': -panebot_emission,
            'botpane_othersideabs': botwall_emission * e_pane,
            'botpane_panereflabs': panebot_emission * r_surf_walls,
            'vf': self.vfs['botpane->botwall']
        }

        # 2 - towards surf
        surf_emission = e_surf_walls * sb__W_m2K4 * (TsurfK ** 4) * self.surf_pane_area__m2
        # we emit out, and receive surf emssion that we absorb
        Qbotpane_surf = -panebot_emission + surf_emission * e_pane
        # and one refl off surf absorbed fully
        Qbotpane_surf += panebot_emission * r_surf_walls
        surf_bits = {
            'botpane_emission': -panebot_emission,
            'botpane_othersideabs': surf_emission * e_pane,
            'botpane_panereflabs': panebot_emission * r_surf_walls,
            'vf': self.vfs['botpane->surf']
        }

        # Top side:
        # 1 - towards top walls (at air temp)
        TtopwallC = Ta__C
        TtopwallK = TtopwallC + 273.15
        topwall_emission = e_surf_walls * sb__W_m2K4 * (TtopwallK ** 4) * self.surf_pane_area__m2
        # receive from topwall as we absorb
        Qtoppane_topwall = -panetop_emission + topwall_emission * e_pane

        # and absorb one sky reflection as much as we absorb
        sky_emission = sky__W_m2 * self.surf_pane_area__m2
        Qtoppane_topwall += sky_emission * r_surf_walls * e_pane
        topwall_bits = {
            'toppane_emission': -panetop_emission,
            'toppane_othersideabs': topwall_emission * e_pane,
            'toppane_skyreflabs': sky_emission * r_surf_walls * e_pane,
            # 'toppane_skyreflabs': 0,
            'vf': self.vfs['toppane->topwall']
        }

        # 2 - towards sky
        Qtoppane_sky = -panetop_emission + sky_emission * e_pane
        # no refls we just get it direct
        sky_bits = {
            'toppane_emission': -panetop_emission,
            'toppane_othersideabs': sky_emission * e_pane,
            'toppane_skyreflabs': 0,
            'vf': self.vfs['toppane->sky']
        }

        # scale by view factors and add up
        Qbotpane_all = (
            self.vfs['botpane->botwall'] * Qbotpane_botwall +
            self.vfs['botpane->surf'] * Qbotpane_surf
        )
        Qtoppane_all = (
            self.vfs['toppane->topwall'] * Qtoppane_topwall +
            self.vfs['toppane->sky'] * Qtoppane_sky
        )
        Qpane_all = Qbotpane_all + Qtoppane_all

        # def linebit(Q, vf):
        #     return [
        #         float("%.2f" % (Q/self.surf_pane_area__m2)),
        #         float("%.4f" % vf),
        #         float("%.2f" % (Q*vf/self.surf_pane_area__m2)),
        #     ]

        bits.update({
            # '- Qbotpane_ir_botwall ': linebit(Qbotpane_botwall, self.vfs['botpane->botwall']),
            # '- Qbotpane_ir_surf    ': linebit(Qbotpane_surf, self.vfs['botpane->surf']),
            # '- Qtoppane_ir_topwall ': linebit(Qtoppane_topwall, self.vfs['toppane->topwall']),
            # '- Qtoppane_ir_sky     ': linebit(Qtoppane_sky, self.vfs['toppane->sky']),
            '- - (Qbotpane_ir_emit)': sum(bit['botpane_emission']*bit['vf'] for bit in [botwall_bits, surf_bits]),
            '- - (Qbotpane_ir_othersideabs)': sum(bit['botpane_othersideabs']*bit['vf'] for bit in [botwall_bits, surf_bits]),
            '- - (Qbotpane_ir_panereflabs)': sum(bit['botpane_panereflabs']*bit['vf'] for bit in [botwall_bits, surf_bits]),
            '- (Qbotpane_ir_total)': float("%.2f" % (Qbotpane_all)),
            '- - (Qtoppane_ir_emit)': sum(bit['toppane_emission']*bit['vf'] for bit in [topwall_bits, sky_bits]),
            '- - (Qtoppane_ir_othersideabs)': sum(bit['toppane_othersideabs']*bit['vf'] for bit in [topwall_bits, sky_bits]),
            '- - (Qtoppane_ir_skyreflabs)': sum(bit['toppane_skyreflabs']*bit['vf'] for bit in [topwall_bits, sky_bits]),
            '- (Qtoppane_ir_total)': float("%.2f" % (Qtoppane_all)),
            'Qpane_ir': Qpane_all,
        })

        return Qpane_all, bits

    def Qsurf_from_solar(self, insolation__W_m2):
        # no view factor scaling from the sun, since it's a point source from afar
        pane_key = 'panetop' if self.two_layer_pane else 'pane'

        pane_trans = self.pieces[pane_key].solar_trans
        surf_abs = self.pieces['surf'].solar_abs
        return (
            insolation__W_m2 * self.surf_pane_area__m2 * pane_trans * surf_abs
        )

    def Qpane_from_solar(self, insolation__W_m2):
        # no view factor scaling from the sun, since it's a point source from afar
        pane_key = 'panetop' if self.two_layer_pane else 'pane'

        pane_abs = self.pieces[pane_key].solar_abs

        pane_trans = self.pieces[pane_key].solar_trans
        surf_refl = self.pieces['surf'].solar_refl

        # also fully absorbs 1 refl off surface
        return (
            insolation__W_m2 * self.surf_pane_area__m2 * pane_abs +
            insolation__W_m2 * self.surf_pane_area__m2 * pane_trans * surf_refl
        )

    def step(self, Ta__C, insolation__W_m2, sky__W_m2):
        """Do one step of sim, with the given air temperature and insolation and sky backradiation"""
        # we have a few interfaces
        # want to calculate Q for the surface and the pane
        # positive = it gains heat, negative = it loses heat
        Qsurf__W = 0
        Qpanebot__W = 0
        Qpanetop__W = 0

        self.latest__W = {}

        # Surface:

        # non-radiative:
        # cond through cylinder
        self.latest__W['- Qsurf_cond_cyl'] = self.Q_conduction_surf_cylinder__W(Ta__C=Ta__C)
        Qsurf__W += self.latest__W['- Qsurf_cond_cyl']
        # conduction_surf_pane (through air)
        # (conduction-only justified by low Raleigh number <<< 1708)
        self.latest__W['- Qsurf_cond_pane'] = self.Q_conduction_surf_pane__W()

        # cylinder would gain but we presume infinite sink keeping it such that
        # sides are at air temp

        # surface gains whatever bottom pane loses (& vice versa)
        Qsurf__W += self.latest__W['- Qsurf_cond_pane']

        self.latest__W['Qsurf_cond'] = (
            self.latest__W['- Qsurf_cond_cyl'] +
            self.latest__W['- Qsurf_cond_pane']
        )

        # radiative infrared exchanges
        # surface <-> pane, walls & sky
        updW, upd_info = self.Qir_surf__W(sky__W_m2=sky__W_m2, Ta__C=Ta__C)
        Qsurf__W += updW
        self.latest__W.update(upd_info)

        # radiative solar:
        # surface: absorption of w/e transmitted through pane
        self.latest__W['Qsurf_solar'] = self.Qsurf_from_solar(insolation__W_m2)
        Qsurf__W += self.latest__W['Qsurf_solar']

        self.latest__W['Qsurf_total'] = Qsurf__W

        # -----------
        # Pane:

        self.latest__W['-----'] = ''

        # from surface, it gains whatever surface loses (& vice versa)
        Qpanebot__W -= self.latest__W['- Qsurf_cond_pane']

        # convection_pane_air (from top of pane)
        self.latest__W['Qpane_cond_surf'] = -self.latest__W['- Qsurf_cond_pane']
        self.latest__W['Qpane_conv_air'] = self.Q_convection_pane_air__W(Ta__C=Ta__C)
        Qpanetop__W += self.latest__W['Qpane_conv_air']

        # pane <-> surface, walls & sky
        # this is evenly divided along the pane bottom and top
        updW, upd_info = self.Qir_pane__W(sky__W_m2=sky__W_m2, Ta__C=Ta__C)
        Qpanebot__W += updW/2
        Qpanetop__W += updW/2
        self.latest__W.update(upd_info)

        # pane: absorption of w/e is not (transmitted or reflected) through the pane, again distributed
        # among both objects
        self.latest__W['Qpane_solar'] = self.Qpane_from_solar(insolation__W_m2)
        Qpanebot__W += self.latest__W['Qpane_solar']/2
        Qpanetop__W += self.latest__W['Qpane_solar']/2

        # conduction from pane bottom to pane top in 2-layer case
        if self.two_layer_pane:
            self.latest__W['Qpanebot_cond_top__W'] = self.Qpanebot_cond_top__W()
            self.latest__W['Qpanetop_cond_bot__W'] = -self.latest__W['Qpanebot_cond_top__W']
            Qpanebot__W += self.latest__W['Qpanebot_cond_top__W']
            Qpanetop__W += self.latest__W['Qpanetop_cond_bot__W']
            self.latest__W['Qpanebot_total'] = Qpanebot__W
            self.latest__W['Qpanetop_total'] = Qpanetop__W

        self.latest__W['Qpane_total'] = Qpanebot__W + Qpanetop__W

        self.latest__W['------'] = ''
        pane_key = 'panetop' if self.two_layer_pane else 'pane'
        self.latest__W.update({
            'surf_solar_tra': [
                self.pieces['surf'].solar_trans,
                self.pieces['surf'].solar_refl,
                self.pieces['surf'].solar_abs,
            ],
            'pane_solar_tra': [
                self.pieces[pane_key].solar_trans,
                self.pieces[pane_key].solar_refl,
                self.pieces[pane_key].solar_abs,
            ],
            'pane_ir_tra': [
                self.pieces[pane_key].ir_trans,
                self.pieces[pane_key].ir_refl,
                self.pieces[pane_key].ir_abs,
            ],
        })

        # ------------------
        # Watts are J/S, calculate J based on time step size
        surfgain__J = Qsurf__W * self.step__s
        panebotgain__J = Qpanebot__W * self.step__s
        panetopgain__J = Qpanetop__W * self.step__s

        # update components, this updates their temps too
        self.pieces['surf'].update_ie(surfgain__J)
        if self.two_layer_pane:
            self.pieces['panebot'].update_ie(panebotgain__J)
            self.pieces['panetop'].update_ie(panetopgain__J)
        else:
            self.pieces['pane'].update_ie(panebotgain__J + panetopgain__J)

        self.time__s += self.step__s
        self.n_steps += 1

        any_change = (
            abs(surfgain__J) > 0.000001
            or abs(panebotgain__J) > 0.000001
            or abs(panetopgain__J) > 0.000001
        )

        return any_change

    def fmt(self, caf2_surf_C=None):
        return """\
T=%.0fs
PaneTop: %.2f ºC
PaneBot: %.2f ºC
Surf: %.2f ºC%s
Flows (W/m^2):
%s\
""" % (
            self.time__s,
            self.pieces['panetop' if self.two_layer_pane else 'pane'].T__C,
            self.pieces['panebot' if self.two_layer_pane else 'pane'].T__C,
            self.pieces['surf'].T__C,
            (" (%+.2f vs. CaF2)" % (
                self.pieces['surf'].T__C - caf2_surf_C
            )) if caf2_surf_C is not None else '',
            "\n".join(
                "%s: %s" % (
                    k,
                    ("%.2f"%(v/(0 or self.surf_pane_area__m2))) if isinstance(v, (int, float)) else str(v),
                )
                for k, v in self.latest__W.items()
            )
        )


def run_steady_state(
        Ta__C,
        insolation__W_m2, sky__W_m2,
        pane_h__W_m2K, cylinder__S_m,
        two_layer_pane=False,
        fixed_temp_Cs: dict[str, float] = None,
        verbose=0,
        materials_to_run=None,  # specify which pane materials to run
):
    fixed_temp_Cs = fixed_temp_Cs or {}

    # Default to all three materials if not specified
    if materials_to_run is None:
        materials_to_run = ['borosilicate', 'sapphire', 'caf2']

    cls = ModelSurfPane

    mat_insts = {
        mat: cls(
            Tstart__C=Ta__C,
            pane_h__W_m2K=pane_h__W_m2K,
            cylinder__S_m=cylinder__S_m,
            surf_material=materials['black_pla'],
            pane_material=materials[mat],
            two_layer_pane=two_layer_pane,
        )
        for mat in materials_to_run
    }

    # extract individual models if they exist
    boro = mat_insts.get('borosilicate')
    sapph = mat_insts.get('sapphire')
    caf2 = mat_insts.get('caf2')

    # Track temperatures for stability check
    prev_temps = {}
    model_list = []
    if boro:
        model_list.append(('boro', boro))
    if sapph:
        model_list.append(('sapph', sapph))
    if caf2:
        model_list.append(('caf2', caf2))

    for model_name, model in model_list:
        prev_temps[model_name] = {
            k: model.pieces[k].T__C
            for k in model.pieces.keys()
        }

    # Track temperature history for repetition detection
    # Store (time, temperature) tuples for each model and piece
    temp_history = {}
    for model_name, model in model_list:
        temp_history[model_name] = {
            k: deque()  # Will store (time_s, temp_C) tuples
            for k in model.pieces.keys()
        }

    first_stable_min = None
    minute = 0
    second = 0
    print_every_mins = 1
    temp_threshold = 0.0001  # Temperature change threshold in Celsius
    repetition_detected = False

    def printit():
        headers = []
        fmts = []
        caf2_surf_C = caf2.pieces['surf'].T__C if caf2 else None
        if boro:
            headers.append("boro")
            fmts.append(boro.fmt(caf2_surf_C=caf2_surf_C))
        if sapph:
            headers.append("sapph")
            fmts.append(sapph.fmt(caf2_surf_C=caf2_surf_C))
        if caf2:
            headers.append("caf2")
            fmts.append(caf2.fmt())
        if headers:
            print(tabulate.tabulate([
                headers,
                fmts,
            ]))

    while True:
        for model in mat_insts.values():
            model.step(
                Ta__C=Ta__C,
                insolation__W_m2=insolation__W_m2,
                sky__W_m2=sky__W_m2,
            )
            for piece, fixedC in fixed_temp_Cs.items():
                if fixedC is None:
                    continue

                if piece not in model.pieces:
                    continue

                model.pieces[piece].initial_T__K = fixedC + 273.15
                model.pieces[piece].ie__J = 0
                model.pieces[piece].initialize()

        # Record temperatures after each step
        # Use the first model's time (all should have same time)
        current_time = list(mat_insts.values())[0].time__s if mat_insts else 0
        for model_name, model in model_list:
            # Record temperature for each piece dynamically
            for piece_name in model.pieces.keys():
                piece_temp = round(model.pieces[piece_name].T__C, 2)  # Round to 0.01°C precision
                
                # Add to history
                temp_history[model_name][piece_name].append((current_time, piece_temp))
                
                # Remove old entries (older than 5 minutes)
                while (temp_history[model_name][piece_name] and 
                       temp_history[model_name][piece_name][0][0] < current_time - 300):
                    temp_history[model_name][piece_name].popleft()

        if current_time > second:
            second += 1
            if int(verbose) >= 3:
                printit()

        # Check for new minute
        if current_time // 60 > minute:
            minute += 1

            # Check temperature stability
            temps_stable = True
            for model_name, model in model_list:
                # Check stability for all pieces
                for piece_name in model.pieces.keys():
                    piece_temp = model.pieces[piece_name].T__C
                    piece_delta = abs(piece_temp - prev_temps[model_name][piece_name])
                    
                    if piece_delta > temp_threshold:
                        temps_stable = False
                    
                    # Update previous temperatures
                    prev_temps[model_name][piece_name] = piece_temp

            # format models side-by-side with header
            if int(verbose) >= 2 and print_every_mins and minute % print_every_mins == 0:
                printit()

            if temps_stable:
                if not first_stable_min:
                    first_stable_min = minute
                else:
                    if minute - first_stable_min >= 5:
                        if int(verbose) >= 2:
                            print(
                                f"\nReached steady state: temperatures stable (< {temp_threshold}°C change) for 5 minutes")
                        break
            else:
                first_stable_min = None

            # Also break if repetition was detected
            if repetition_detected:
                break

    if int(verbose) >= 1:
        printit()

    # Return models in the order they were requested
    return [mat_insts.get(mat) for mat in materials_to_run]


def find_material_column(df_columns, material_prefix):
    """Find column that starts with the given material prefix.
    
    Args:
        df_columns: List of column names from the dataframe
        material_prefix: Material name to search for (e.g. 'CaF2', 'Sapph', 'Boro')
    
    Returns:
        Column name if found, None otherwise
    """
    # Normalize the prefix for comparison
    prefix_lower = material_prefix.lower()
    
    # Look for exact prefix match (case-insensitive)
    for col in df_columns:
        if col.lower().startswith(prefix_lower + '/'):
            return col
    
    # Try alternate names
    alternates = {
        'caf2': ['caf2', 'calcium fluoride'],
        'sapph': ['sapph', 'sapphire'],  
        'boro': ['boro', 'borosilicate']
    }
    
    if prefix_lower in alternates:
        for alt in alternates[prefix_lower]:
            for col in df_columns:
                if col.lower().startswith(alt + '/'):
                    return col
    
    return None


def add_condition_backgrounds(fig, df):
    """
    Add background shading for condition ranges using weighted color averaging.

    Args:
        fig: Plotly figure
        df: DataFrame with datetime column
        wrap: Whether to wrap times to 24-hour format
        file_condition_ranges: Optional dict of condition ranges per file for filtering
        row_files_map: Optional dict mapping row numbers to files for selective filtering
    """
    # Aggregate conditions per minute
    minute_aggregation = main_exp_plots.aggregate_conditions_per_minute(
        main_exp_plots.CONDITION_RANGES, df, wrap=False,
    )

    df_min_dt = df['datetime'].min()
    df_max_dt = df['datetime'].max()

    # Sort minutes to find contiguous ranges
    sorted_minutes = sorted(minute_aggregation.keys())

    # Group contiguous minutes with same condition mix
    i = 0
    while i < len(sorted_minutes):
        start_minute = sorted_minutes[i]
        if start_minute < df_min_dt:
            i += 1
            continue

        if start_minute > df_max_dt:
            break

        current_conditions = minute_aggregation[start_minute]['conditions']

        # Find end of contiguous range with same conditions
        j = i + 1
        while j < len(sorted_minutes):
            next_minute = sorted_minutes[j]
            if next_minute > df_max_dt:
                break
            next_conditions = minute_aggregation[next_minute]['conditions']

            # Check if next minute is contiguous and has same conditions
            # For non-wrapped times, check if minutes are consecutive
            time_diff = (next_minute - sorted_minutes[j - 1]).total_seconds() / 60
            if time_diff != 1 or current_conditions != next_conditions:
                break
            j += 1

        # Draw vrect for this contiguous range
        end_minute = sorted_minutes[j - 1]

        # Calculate weighted average color
        colors_with_weights = []
        for condition, count in current_conditions.items():
            if condition in graph_config.CONDITION_COLORS:
                color_config = graph_config.CONDITION_COLORS[condition]
                colors_with_weights.append((color_config['color'], color_config['opacity'], count))

        if colors_with_weights:
            avg_color, avg_opacity = graph_config.average_colors_weighted(colors_with_weights)

            # Add vrect with averaged color
            vrect_config = {'color': avg_color, 'opacity': 1.0}

            # Calculate actual end time (add 1 minute to include the full minute)
            end_time = end_minute + pd.Timedelta(minutes=1)

            # Draw on specified rows
            main_exp_plots.add_vrect_to_specific_rows(fig, start_minute, end_time, vrect_config, [1, 2, 3])

        i = j


def plot_reality_trace(real_df, modeled_df, infilled_indices=None):
    """Plot real temperature data and modeled temperatures together.
    
    Args:
        real_df: DataFrame with real temperature data and environmental conditions
        modeled_df: DataFrame with modeled surface and pane temperatures
        infilled_indices: Dictionary with 'solar' and 'ir_sky' keys containing indices of infilled data
    """
    # Color configuration from graph_config.py style
    colors = {
        'air': '#ff00af',
        'CaF2': '#00b7ef',
        'Sapph': '#6f3198', 
        'Boro': '#ff7e00',
        'solar': '#ffc20e',
        'ir_sky': '#4d6df3',
    }

    # Create figure with subplots - 3 rows: combined temps/radiation, relative temps, modeled vs measured diffs
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Surface Temperatures & Environmental Conditions', 'Relative Temperature Differences', 'Modeled vs Measured Differences'),
        vertical_spacing=0.08,  # Increased to prevent title/axis overlap
        shared_xaxes=True,
        row_heights=[1, 1, 1],  # All rows same size
        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )

    # Add condition backgrounds to all rows
    # Aggregate conditions per minute for the real data
    # Load condition ranges from CSV files
    add_condition_backgrounds(fig, real_df)

    # Row 1: Combined temperatures and environmental conditions
    # Air temperature (primary y-axis, with temperatures)
    fig.add_trace(
        go.Scatter(
            x=real_df['datetime'],
            y=real_df['air'],
            mode='lines',
            name='Air Temp',
            line=dict(color=colors['air'], width=2),
            legendgroup='env',
        ),
        row=1, col=1, secondary_y=False
    )
    
    # Add infilled air data overlay if provided
    if infilled_indices and infilled_indices.get('air'):
        # Split infilled indices into consecutive chunks
        air_indices = sorted(infilled_indices['air'])
        if air_indices:
            # Group consecutive indices
            chunks = []
            current_chunk = [air_indices[0]]
            
            for i in range(1, len(air_indices)):
                if air_indices[i] == air_indices[i-1] + 1:
                    current_chunk.append(air_indices[i])
                else:
                    chunks.append(current_chunk)
                    current_chunk = [air_indices[i]]
            chunks.append(current_chunk)
            
            # Create a trace for each consecutive chunk
            for i, chunk in enumerate(chunks):
                infilled_air = real_df.loc[chunk]
                if not infilled_air.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=infilled_air['datetime'],
                            y=infilled_air['air'],
                            mode='lines',
                            name='Air (infilled)' if i == 0 else None,  # Only label first trace
                            line=dict(color='red', width=2, dash='solid'),  # Solid red for air
                            legendgroup='air_infilled',
                            showlegend=(i == 0),  # Only show first in legend
                        ),
                        row=1, col=1, secondary_y=False
                    )
    
    # Solar radiation (secondary y-axis)
    fig.add_trace(
        go.Scatter(
            x=real_df['datetime'],
            y=real_df['solar'],
            mode='lines',
            name='Solar',
            line=dict(color=colors['solar'], width=2, dash='dash'),
            legendgroup='env',
        ),
        row=1, col=1, secondary_y=True
    )
    
    # Add infilled solar data overlay if provided
    if infilled_indices and infilled_indices.get('solar'):
        # Split infilled indices into consecutive chunks
        solar_indices = sorted(infilled_indices['solar'])
        if solar_indices:
            # Group consecutive indices
            chunks = []
            current_chunk = [solar_indices[0]]
            
            for i in range(1, len(solar_indices)):
                if solar_indices[i] == solar_indices[i-1] + 1:
                    current_chunk.append(solar_indices[i])
                else:
                    chunks.append(current_chunk)
                    current_chunk = [solar_indices[i]]
            chunks.append(current_chunk)
            
            # Create a trace for each consecutive chunk
            for i, chunk in enumerate(chunks):
                infilled_solar = real_df.loc[chunk]
                if not infilled_solar.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=infilled_solar['datetime'],
                            y=infilled_solar['solar'],
                            mode='lines',
                            name='Solar (infilled)' if i == 0 else None,  # Only label first trace
                            line=dict(color='red', width=2, dash='dash'),
                            legendgroup='env_infilled',
                            showlegend=(i == 0),  # Only show first in legend
                        ),
                        row=1, col=1, secondary_y=True
                    )
    
    # Sky radiation (secondary y-axis)
    fig.add_trace(
        go.Scatter(
            x=real_df['datetime'],
            y=real_df['ir_sky'],
            mode='lines',
            name='Sky IR',
            line=dict(color=colors['ir_sky'], width=2, dash='dash'),
            legendgroup='env',
        ),
        row=1, col=1, secondary_y=True
    )
    
    # Add infilled ir_sky data overlay if provided
    if infilled_indices and infilled_indices.get('ir_sky'):
        # Split infilled indices into consecutive chunks
        ir_indices = sorted(infilled_indices['ir_sky'])
        if ir_indices:
            # Group consecutive indices
            chunks = []
            current_chunk = [ir_indices[0]]
            
            for i in range(1, len(ir_indices)):
                if ir_indices[i] == ir_indices[i-1] + 1:
                    current_chunk.append(ir_indices[i])
                else:
                    chunks.append(current_chunk)
                    current_chunk = [ir_indices[i]]
            chunks.append(current_chunk)
            
            # Create a trace for each consecutive chunk
            for i, chunk in enumerate(chunks):
                infilled_ir = real_df.loc[chunk]
                if not infilled_ir.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=infilled_ir['datetime'],
                            y=infilled_ir['ir_sky'],
                            mode='lines',
                            name='Sky IR (infilled)' if i == 0 else None,  # Only label first trace
                            line=dict(color='red', width=2, dash='dash'),
                            legendgroup='ir_infilled',
                            showlegend=(i == 0),  # Only show first in legend
                        ),
                        row=1, col=1, secondary_y=True
                    )
    
    # Add surface and pane temperatures to Row 1
    # Process each material with proper legend grouping
    for material, col_prefix, surf_col, pane_col in [
        ('CaF2', 'CaF2/', 'caf2_surf', 'caf2_pane'),
        ('Sapph', 'Sapph/', 'sapph_surf', 'sapph_pane'), 
        ('Boro', 'Boro/', 'boro_surf', 'boro_pane')
    ]:
        # Find the real data column for this material
        real_col = None
        for col in real_df.columns:
            if col.startswith(col_prefix):
                real_col = col
                break
        
        # 1. Real surface temperature (solid line)
        if real_col:
            fig.add_trace(
                go.Scatter(
                    x=real_df['datetime'],
                    y=real_df[real_col],
                    mode='lines',
                    name=f'{material} Real',
                    line=dict(color=colors[material], width=2.5),
                    legendgroup=material,  # Group by material
                    legendgrouptitle_text=material,
                ),
                row=1, col=1, secondary_y=False
            )
        
        fig.add_trace(
            go.Scatter(
                x=modeled_df['datetime'],
                y=modeled_df[surf_col],
                mode='lines',
                name=f'{material} Model',
                line=dict(color=colors[material], width=2.5, dash='dot'),
                legendgroup=material,
            ),
            row=1, col=1, secondary_y=False
        )
        #
        # # 3. Pane temperature (dash-dot line, semi-transparent)
        # fig.add_trace(
        #     go.Scatter(
        #         x=modeled_df['datetime'],
        #         y=modeled_df[pane_col],
        #         mode='lines',
        #         name=f'{material} Pane',
        #         line=dict(color=colors[material], width=2, dash='dashdot'),
        #         opacity=0.7,
        #         legendgroup=material,
        #     ),
        #     row=2, col=1
        # )
    
    # Row 2: Relative temperatures (differences between materials)
    # Calculate Boro vs CaF2 and Sapph vs CaF2 for both real and modeled data
    
    # Find the columns
    caf2_real_col = None
    boro_real_col = None
    sapph_real_col = None
    for col in real_df.columns:
        if col.startswith('CaF2/'):
            caf2_real_col = col
        elif col.startswith('Boro/'):
            boro_real_col = col
        elif col.startswith('Sapph/'):
            sapph_real_col = col
    
    # Real data relative temperatures
    if boro_real_col and caf2_real_col:
        boro_vs_caf2_real = real_df[boro_real_col] - real_df[caf2_real_col]
        fig.add_trace(
            go.Scatter(
                x=real_df['datetime'],
                y=boro_vs_caf2_real,
                mode='lines',
                name='Boro-CaF2 Real',
                line=dict(color=colors['Boro'], width=2.5),
                legendgroup='relative_real',
            ),
            row=2, col=1
        )
    
    if sapph_real_col and caf2_real_col:
        sapph_vs_caf2_real = real_df[sapph_real_col] - real_df[caf2_real_col]
        fig.add_trace(
            go.Scatter(
                x=real_df['datetime'],
                y=sapph_vs_caf2_real,
                mode='lines',
                name='Sapph-CaF2 Real',
                line=dict(color=colors['Sapph'], width=2.5),
                legendgroup='relative_real',
            ),
            row=2, col=1
        )
    
    # Modeled data relative temperatures (use pre-computed values)
    fig.add_trace(
        go.Scatter(
            x=modeled_df['datetime'],
            y=modeled_df['boro_vs_caf2'],
            mode='lines',
            name='Boro-CaF2 Model',
            line=dict(color=colors['Boro'], width=2.5, dash='dot'),
            legendgroup='relative_model',
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=modeled_df['datetime'],
            y=modeled_df['sapph_vs_caf2'],
            mode='lines',
            name='Sapph-CaF2 Model',
            line=dict(color=colors['Sapph'], width=2.5, dash='dot'),
            legendgroup='relative_model',
        ),
        row=2, col=1
    )
    
    # Add horizontal line at y=0 for reference in row 2
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=2, col=1)
    
    # Row 3: Temperature differences (Modeled - Measured)
    for material, col_prefix, surf_col, _ in [
        ('CaF2', 'CaF2/', 'caf2_surf', 'caf2_pane'),
        ('Sapph', 'Sapph/', 'sapph_surf', 'sapph_pane'), 
        ('Boro', 'Boro/', 'boro_surf', 'boro_pane')
    ]:
        # Find the real data column for this material
        real_col = None
        for col in real_df.columns:
            if col.startswith(col_prefix):
                real_col = col
                break
        
        if real_col and surf_col in modeled_df.columns:
            # Calculate difference (modeled - real)
            # Align the dataframes on datetime for proper comparison
            merged = pd.merge_asof(
                modeled_df[['datetime', surf_col]].sort_values('datetime'),
                real_df[['datetime', real_col]].sort_values('datetime'),
                on='datetime',
                direction='nearest'
            )
            diff = merged[surf_col] - merged[real_col]
            
            fig.add_trace(
                go.Scatter(
                    x=merged['datetime'],
                    y=diff,
                    mode='lines',
                    name=f'{material} Diff',
                    line=dict(color=colors[material], width=2.5),
                    legendgroup=f'{material}_diff',
                    showlegend=True,
                ),
                row=3, col=1
            )
    
    # Add horizontal line at y=0 for reference
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=3, col=1)
    
    # Calculate y-axis range for difference plot (centered around 0 with integer ticks)
    all_diffs = []
    for material, col_prefix, surf_col, _ in [
        ('CaF2', 'CaF2/', 'caf2_surf', 'caf2_pane'),
        ('Sapph', 'Sapph/', 'sapph_surf', 'sapph_pane'), 
        ('Boro', 'Boro/', 'boro_surf', 'boro_pane')
    ]:
        real_col = None
        for col in real_df.columns:
            if col.startswith(col_prefix):
                real_col = col
                break
        if real_col and surf_col in modeled_df.columns:
            merged = pd.merge_asof(
                modeled_df[['datetime', surf_col]].sort_values('datetime'),
                real_df[['datetime', real_col]].sort_values('datetime'),
                on='datetime',
                direction='nearest'
            )
            diff = merged[surf_col] - merged[real_col]
            all_diffs.extend(diff.dropna().tolist())
    
    y_range = 3.5
    
    # Add ranking match indicator rectangles to Row 2
    # Group consecutive True/False values into segments
    segments = []
    current_segment = None
    
    for i, (datetime, match) in enumerate(zip(modeled_df['datetime'], modeled_df['ranking_match'])):
        if current_segment is None:
            # Start new segment
            current_segment = {'start': datetime, 'end': datetime, 'match': match}
        elif current_segment['match'] == match:
            # Continue current segment
            current_segment['end'] = datetime
        else:
            # End current segment and start new one
            segments.append(current_segment)
            current_segment = {'start': datetime, 'end': datetime, 'match': match}
    
    # Don't forget the last segment
    if current_segment is not None:
        segments.append(current_segment)
    
    # Add vertical rectangles for each segment in row 2 (relative temperatures)
    y_range_row2 = graph_config.TEMP_DIFF_Y_RANGE
    for segment in segments:
        color = 'rgba(0, 255, 0, 0.8)' if segment['match'] else 'rgba(255, 0, 0, 0.8)'
        fig.add_shape(
            type='rect',
            x0=segment['start'],
            x1=segment['end'],
            y0=y_range_row2[1] - 0.4,  # Position at top of chart
            y1=y_range_row2[1],
            fillcolor=color,
            layer='below',
            line_width=0,
            row=2, col=1,
        )
    
    # Update axes with graph_config font sizes
    fig.update_xaxes(title_text="Time", row=3, col=1, 
                     title_font=dict(size=graph_config.AXISFONT_SIZE),
                     tickfont=dict(size=graph_config.TICKFONT_SIZE),
                     showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    # Row 1: Temperature and radiation axes
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1, secondary_y=False,
                     title_font=dict(size=graph_config.AXISFONT_SIZE),
                     tickfont=dict(size=graph_config.TICKFONT_SIZE),
                     range=graph_config.TEMP_Y_RANGE,
                     showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(title_text="Radiation (W/m²)", row=1, col=1, secondary_y=True,
                     title_font=dict(size=graph_config.AXISFONT_SIZE), 
                     tickfont=dict(size=graph_config.TICKFONT_SIZE),
                     range=graph_config.RADIATION_Y_RANGE,
                     showgrid=False)
    
    # Row 2: Relative temperature differences
    tick_vals = np.arange(-5, 5, graph_config.TEMP_DIFF_TICK_INTERVAL)
    fig.update_yaxes(
        title_text="ΔT (°C)", 
        row=2, col=1,
        title_font=dict(size=graph_config.AXISFONT_SIZE-2),
        tickfont=dict(size=graph_config.TICKFONT_SIZE),
        range=graph_config.TEMP_DIFF_Y_RANGE,
        tickvals=tick_vals,
        ticktext=[f"{x:.1f}" for x in tick_vals],
        showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='gray'
    )
    
    # Row 3: Modeled vs measured differences
    fig.update_yaxes(
        title_text="Model - Measured (°C)",
        row=3, col=1,
        title_font=dict(size=graph_config.AXISFONT_SIZE-2),
        tickfont=dict(size=graph_config.TICKFONT_SIZE),
        range=[-y_range, y_range],
        dtick=1,  # Integer ticks
        showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='gray'
    )
    
    # Get date range for title
    start_date = real_df['datetime'].min().strftime('%Y-%m-%d %H:%M')
    end_date = real_df['datetime'].max().strftime('%H:%M')
    
    # Update overall layout with graph_config settings
    fig.update_layout(
        title={
            'text': f"Real vs Modeled Temperatures - Greenhouse Experiment ({start_date} to {end_date})",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': graph_config.TITLE_FONT_SIZE, 'color': 'black'}
        },
        height=400 * sum([1, 1, 1]),  # 1800px total, 600px per row
        width=graph_config.PLOT_WIDTH,
        hovermode='x unified',
        showlegend=True,
        font=dict(family="Computer Modern, Times New Roman, serif", color="black"),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            font=dict(size=graph_config.LEGEND_FONT_SIZE),
            tracegroupgap=10,
            groupclick="toggleitem"  # Make legend items individually clickable
        ),
        plot_bgcolor='rgba(248,249,250,0.8)',
        paper_bgcolor='white',
    )
    
    # Update subplot titles font size
    fig.update_annotations(font_size=graph_config.SUBTITLE_FONT_SIZE)
    
    # Format x-axes to show time nicely for all rows
    for row in [1, 2, 3]:
        fig.update_xaxes(tickformat='%H:%M', 
                         tickfont=dict(size=graph_config.TICKFONT_SIZE),
                         mirror="allticks", showticklabels=True, row=row, col=1,
                         showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    # Create HTML and open in browser
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        html_content = fig.to_html(
            include_plotlyjs='cdn',
            config={
                'displayModeBar': True,
                'scrollZoom': True,
                'doubleClick': 'reset+autosize',
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'reality_trace',
                    'height': 800,
                    'width': 1400,
                    'scale': 2
                },
            }
        )
        
        # Add some styling
        enhanced_html = html_content.replace(
            '<head>',
            '''<head>
            <style>
                body { 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    margin: 0;
                    padding: 20px;
                    font-family: 'Arial', sans-serif;
                }
                .plotly-graph-div {
                    border-radius: 15px;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                    background: white;
                    padding: 10px;
                }
            </style>'''
        )
        
        f.write(enhanced_html)
        temp_file = f.name
    
    webbrowser.open(f'file://{temp_file}')
    print(f"Plot saved and opened: {temp_file}")
    return temp_file


class CmdLine:
    def run_ss(
        self,
        scenario,
        fix_pane=None, fix_surf=None,
        two_layer_pane=False,
        verbose=False,
        counterfactuals=None,
    ):
        if scenario == 'peak':
            Ta__C = 30.2
            insolation__W_m2 = 855.7
            sky__W_m2 = 371.8  # measured97
            pane_h__W_m2K = 30
        elif scenario == 'night':
            Ta__C = 16.2
            insolation__W_m2 = 0
            sky__W_m2 = 324.4  # measured97
            pane_h__W_m2K = 20
        else:
            raise NotImplementedError()

        # counterfactuals parse as comma-separaetd strings list
        if counterfactuals is None:
            counterfactuals = []
        elif isinstance(counterfactuals, tuple):
            counterfactuals = [cf.strip().lower() for cf in counterfactuals]
        elif isinstance(counterfactuals, str):
            counterfactuals = [cf.strip().lower() for cf in counterfactuals.split(',')]

        for cf in counterfactuals:
            if cf == 'equal_solar':
                # set all their solar_tra to borosilicates
                print("Counterfactual: setting all solar_tra to borosilicate values")
                materials['sapphire']['solar_tra'] = materials['borosilicate']['solar_tra']
                materials['caf2']['solar_tra'] = materials['borosilicate']['solar_tra']
                continue

            if cf == 'equal_solar_pane_abs':
                # set all their solar_tra to borosilicates
                print("Counterfactual: setting all solar properties such that pane solar absorbed is equal")
                _ = 'x'
                materials['sapphire']['solar_tra'] = [0.832, _, 0.07865]
                materials['caf2']['solar_tra'] = [0.914, _, 0.07455]
                materials['sapphire']['solar_tra'][1] = 1 - (materials['sapphire']['solar_tra'][0] + materials['sapphire']['solar_tra'][2])
                materials['caf2']['solar_tra'][1] = 1 - (materials['caf2']['solar_tra'][0] + materials['caf2']['solar_tra'][2])
                materials['sapphire']['solar_tra'][1] = float('%.7f' % materials['sapphire']['solar_tra'][1])
                materials['caf2']['solar_tra'][1] = float('%.7f' % materials['caf2']['solar_tra'][1])
                assert abs(sum(materials['sapphire']['solar_tra']) - 1.0) < 1e-6
                assert abs(sum(materials['caf2']['solar_tra']) - 1.0) < 1e-6
                assert all(x>=0 for x in materials['sapphire']['solar_tra'])
                assert all(x>=0 for x in materials['caf2']['solar_tra'])
                continue

            if cf == 'equal_ir':
                # set all their ir_emis to borosilicates
                print("Counterfactual: setting all ir_tra to borosilicate values")
                materials['sapphire']['ir_tra'] = materials['borosilicate']['ir_tra']
                materials['caf2']['ir_tra'] = materials['borosilicate']['ir_tra']
                continue

            if cf == 'uncoated_boro':
                # take 2% of solar from boro trans to boro refl
                print("Counterfactual: setting borosilicate to uncoated values")
                materials['borosilicate']['solar_tra'][0] -= 0.07
                materials['borosilicate']['solar_tra'][1] += 0.07
                continue

            raise NotImplementedError(f"Unknown counterfactual: {cf}")

        cylinder__S_m = S_cylinder

        boro, sapph, caf2 = run_steady_state(
            Ta__C=Ta__C,
            insolation__W_m2=insolation__W_m2,
            sky__W_m2=sky__W_m2,
            pane_h__W_m2K=pane_h__W_m2K,
            cylinder__S_m=cylinder__S_m,
            fixed_temp_Cs={
                'pane': fix_pane,
                'panetop': fix_pane,
                'panebot': fix_pane,
                'surf': fix_surf,
            },
            two_layer_pane=two_layer_pane,
            verbose=verbose,
        )

        # format models side-by-side with header
        print("Air temp (C):        %.1f" % Ta__C)
        print("Insolation (W/m^2):  %.1f" % insolation__W_m2)
        print("Sky Backrad (W/m^2): %.1f" % sky__W_m2)
        print("Cylinder S (m):      %.4f" % cylinder__S_m)
        print("Pane h (W/m^2K):     %.1f" % pane_h__W_m2K)
        caf2_surf_C = caf2.pieces['surf'].T__C
        fmts = [boro.fmt(caf2_surf_C=caf2_surf_C), sapph.fmt(caf2_surf_C=caf2_surf_C), caf2.fmt()]
        print(tabulate.tabulate([
            ["boro", "sapph", "caf2"],
            fmts,
        ]))

    def trace_reality(
        self,
        csv_file,
        start_time,
        hDay=35, hNight=20,
        S=S_cylinder,  # cylinder shape factor
        plot=True,     # whether to plot the results
        dim_boro=0.109,  # diminution factor for borosilicate at low angles (10.9% reduction)
        dim_sapph=0.054,  # diminution factor for sapphire at low angles (5.4% reduction)
        dim_caf2=0.072,  # diminution factor for CaF2 at low angles (7.2% reduction)
        infill_missing_data=True,  # whether to interpolate missing solar/ir_sky data
        hDayA=None,  # optional override for hDay at position A
        hDayB=None,  # optional override for hDay at position B
        hDayC=None,  # optional override for hDay at position C
        two_layer_pane=False,
    ):
        """Trace real temperature data from CSV file starting at specified time.
        
        Args:
            csv_file: Path to CSV file with temperature data
            start_time: Start time in HH:MM format (e.g. "21:00")
            pane_h__W_m2K: Convection coefficient for pane
            cylinder__S_m: Shape factor for cylinder
            verbose: Verbosity level (0=quiet, 1=basic, 2=detailed)
            plot: Whether to generate and display plot of results
            infill_missing_data: Whether to interpolate missing solar/ir_sky data (default True)
        """
        # Load CSV data
        df = pd.read_csv(csv_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Parse start time and filter data
        start_hour, start_minute = map(int, start_time.split(':'))
        # Get the date from the first row and construct full datetime
        base_date = df['datetime'].iloc[0].date()
        start_datetime = pd.Timestamp(year=base_date.year, month=base_date.month, 
                                       day=base_date.day, hour=start_hour, minute=start_minute)
        
        # Filter to start from specified time
        df = df[df['datetime'] >= start_datetime].copy()
        if len(df) == 0:
            print(f"No data found starting from {start_time}")
            return
        
        # Track infilled data
        infilled_indices = {'solar': [], 'ir_sky': [], 'air': []}
        
        # Infill missing data if requested
        if infill_missing_data:
            # Check for missing solar, ir_sky, and air data
            solar_missing = df['solar'].isna()
            ir_sky_missing = df['ir_sky'].isna()
            air_missing = df['air'].isna()
            
            solar_missing_count = solar_missing.sum()
            ir_sky_missing_count = ir_sky_missing.sum()
            air_missing_count = air_missing.sum()
            
            if solar_missing_count > 0 or ir_sky_missing_count > 0 or air_missing_count > 0:
                print(f"Infilling missing data:")
                if solar_missing_count > 0:
                    print(f"  Solar: {solar_missing_count} missing values")
                    # Track which indices were filled
                    infilled_indices['solar'] = df.index[solar_missing].tolist()
                    # Linear interpolation
                    df['solar'] = df['solar'].interpolate(method='linear', limit_direction='both')
                    
                if ir_sky_missing_count > 0:
                    print(f"  IR Sky: {ir_sky_missing_count} missing values")
                    # Track which indices were filled
                    infilled_indices['ir_sky'] = df.index[ir_sky_missing].tolist()
                    # Linear interpolation
                    df['ir_sky'] = df['ir_sky'].interpolate(method='linear', limit_direction='both')
                    
                if air_missing_count > 0:
                    print(f"  Air: {air_missing_count} missing values")
                    # Track which indices were filled
                    infilled_indices['air'] = df.index[air_missing].tolist()
                    # Linear interpolation
                    df['air'] = df['air'].interpolate(method='linear', limit_direction='both')
                print()
            
        # Find material columns dynamically
        columns = df.columns.tolist()
        caf2_col = find_material_column(columns, 'CaF2')
        sapph_col = find_material_column(columns, 'Sapph')
        boro_col = find_material_column(columns, 'Boro')
        
        # Check that we found all required columns
        missing_cols = []
        if not caf2_col:
            missing_cols.append('CaF2')
        if not sapph_col:
            missing_cols.append('Sapphire')
        if not boro_col:
            missing_cols.append('Borosilicate')
        
        if missing_cols:
            print(f"Error: Could not find columns for materials: {', '.join(missing_cols)}")
            print(f"Available columns: {columns}")
            return
        
        # Extract positions from column headers (e.g., "CaF2/A" -> position A)
        material_positions = {}
        for col, mat_name in [(caf2_col, 'caf2'), (sapph_col, 'sapphire'), (boro_col, 'borosilicate')]:
            if '/' in col:
                position = col.split('/')[1][0]  # Get first character after slash (A, B, or C)
                material_positions[mat_name] = position
        
        # Create mapping of positions to hDay overrides
        position_hDay_overrides = {
            'A': hDayA,
            'B': hDayB, 
            'C': hDayC,
        }
            
        # Get initial values
        first_row = df.iloc[0]
        initial_air_C = first_row['air']
        initial_solar = first_row['solar']
        initial_sky = first_row['ir_sky']
        
        # Initial surface temperatures using dynamic column names
        initial_caf2_surf = first_row[caf2_col]
        initial_sapph_surf = first_row[sapph_col]
        initial_boro_surf = first_row[boro_col]

        initial_night = bool(first_row['night'])
        
        print(f"Starting trace from {first_row['datetime']}")
        print(f"Initial conditions:")
        print(f"  Air temp: {initial_air_C:.1f}°C")
        print(f"  Solar: {initial_solar:.1f} W/m²")
        print(f"  Sky backrad: {initial_sky:.1f} W/m²")
        print(f"  Surface temps - CaF2: {initial_caf2_surf:.1f}°C, Sapph: {initial_sapph_surf:.1f}°C, Boro: {initial_boro_surf:.1f}°C")
        print()
        
        # Create models with initial surface temperatures
        models = {}
        original_solar_tra = {}  # Store original solar transmission values
        diminution_factors = {  # Store diminution factors for each material
            'borosilicate': dim_boro,
            'sapphire': dim_sapph,
            'caf2': dim_caf2,
        }
        pane_key = 'panebot' if two_layer_pane else 'pane'
        for mat_name, mat_key, init_surf_C in [
            ('borosilicate', boro_col, initial_boro_surf),
            ('sapphire', sapph_col, initial_sapph_surf),
            ('caf2', caf2_col, initial_caf2_surf),
        ]:
            # Determine initial hDay value for this material
            initial_hDay_value = hDay
            if not initial_night:
                position = material_positions.get(mat_name)
                if position and position_hDay_overrides.get(position) is not None:
                    initial_hDay_value = position_hDay_overrides[position]
            
            model = ModelSurfPane(
                Tstart__C=initial_air_C,
                pane_h__W_m2K=hNight if initial_night else initial_hDay_value,
                cylinder__S_m=S,
                surf_material=materials['black_pla'],
                pane_material=materials[mat_name],
                two_layer_pane=two_layer_pane
            )
            # Set initial surface temperature
            model.pieces['surf'].initial_T__K = init_surf_C + 273.15
            model.pieces['surf'].ie__J = 0
            model.pieces['surf'].initialize()
            models[mat_name] = model
            
            # Store original solar_tra values for the pane
            original_solar_tra[mat_name] = model.pieces[pane_key].solar_tra.copy()
            
        # Prepare phase: run with fixed surface temps until panes stabilize
        print("Preparing models (bringing panes to steady state with fixed surface temps)...")
        prep_minutes = 0
        prev_pane_temps = {name: model.avg_pane__C() for name, model in models.items()}
        stable_count = 0
        
        while prep_minutes < 60*24:  # Should not take nearly a day, just give it enough time
            # Step each model for 1 minute (60 seconds)
            for _ in range(60):
                for name, model in models.items():
                    # Get the initial surface temp for this material
                    if name == 'borosilicate':
                        fixed_surf_C = initial_boro_surf
                    elif name == 'sapphire':
                        fixed_surf_C = initial_sapph_surf
                    else:  # caf2
                        fixed_surf_C = initial_caf2_surf
                        
                    model.step(
                        Ta__C=initial_air_C,
                        insolation__W_m2=initial_solar,
                        sky__W_m2=initial_sky,
                    )
                    
                    # Fix surface temperature
                    model.pieces['surf'].initial_T__K = fixed_surf_C + 273.15
                    model.pieces['surf'].ie__J = 0
                    model.pieces['surf'].initialize()
            
            prep_minutes += 1
            
            # Check if pane temps are stable
            all_stable = True
            for name, model in models.items():
                current_pane_C = model.avg_pane__C()
                if abs(current_pane_C - prev_pane_temps[name]) > 0.01:
                    all_stable = False
                prev_pane_temps[name] = current_pane_C
                
            if all_stable:
                stable_count += 1
                if stable_count >= 5:
                    print(f"Panes stabilized after {prep_minutes} minutes")
                    break
            else:
                stable_count = 0
        else:
            raise ValueError("Preparation phase did not stabilize within 24 hours")

        print("Preparation complete. Surf/Pane temperatures:")
        for name, model in models.items():
            print(f"  {name:12s}: {model.pieces['surf'].T__C:.2f}/{model.avg_pane__C():.2f} °C")
        print()

        # reset steps taken
        for name, model in models.items():
            model.time__s = 0
            model.n_steps = 0

        # Storage for modeled temperatures - store every second
        modeled_temps = {
            'datetime': [],
            'caf2_surf': [],
            'caf2_pane': [],
            'sapph_surf': [],
            'sapph_pane': [],
            'boro_surf': [],
            'boro_pane': [],
            'ranking_match': [],  # Track if temperature rankings match
            'boro_vs_caf2': [],  # Relative temperature: Boro - CaF2
            'sapph_vs_caf2': [],  # Relative temperature: Sapph - CaF2
        }
        
        # Track the initial simulation time
        initial_time = df.iloc[0]['datetime']
        
        # Main simulation: step through CSV data
        print("Starting trace simulation...")
        print()
        
        # Print header
        header = "Time          Air   Solar SkyBR | Real: CaF2 Sapph Boro | Model: CaF2 Sapph Boro | Delta:  CaF2  Sapph  Boro"
        print(header)
        print("-" * len(header))
        
        # Process each row in the CSV
        for idx, row in df.iterrows():
            # Get environmental conditions
            air_C = row['air']
            solar = row['solar']
            sky = row['ir_sky']
            
            # Get real surface temperatures using dynamic column names
            real_caf2 = row[caf2_col]
            real_sapph = row[sapph_col]
            real_boro = row[boro_col]

            # set proper convection coefficient based on day/night
            is_night = bool(int(row['night']))
            for mat in models:
                if is_night:
                    models[mat].pane_h__W_m2K = hNight
                else:
                    # Check for position-specific hDay override
                    position = material_positions.get(mat)
                    if position and position_hDay_overrides.get(position) is not None:
                        models[mat].pane_h__W_m2K = position_hDay_overrides[position]
                    else:
                        models[mat].pane_h__W_m2K = hDay
            
            # Apply time-based solar transmission adjustments
            current_datetime = row['datetime']
            hour = current_datetime.hour + current_datetime.minute / 60.0
            
            for mat_name, model in models.items():
                # Calculate diminution percentage based on time of day
                dim_factor = diminution_factors[mat_name]
                
                if 12 <= hour <= 15:  # 12pm-3pm: no diminution
                    diminution_pct = 0
                elif hour < 9 or hour >= 18:  # before 9am or after 6pm: full diminution
                    diminution_pct = dim_factor
                elif 9 <= hour < 12:  # 9am-12pm: linear interpolation
                    diminution_pct = dim_factor * (12 - hour) / 3
                elif 15 < hour < 18:  # 3pm-6pm: linear interpolation
                    diminution_pct = dim_factor * (hour - 15) / 3
                else:
                    diminution_pct = 0  # shouldn't happen but be safe
                
                # Apply the diminution to pane's solar transmission
                orig_tra = original_solar_tra[mat_name]
                transmission_decrease = orig_tra[0] * diminution_pct
                
                # Update pane's solar_tra: [transmission, reflection, absorption]
                # handle both 2-layer and 1-layer
                for piece in ['pane', 'panetop', 'panebot']:
                    if piece in model.pieces:
                        model.pieces[piece].solar_tra[0] = orig_tra[0] - transmission_decrease
                        model.pieces[piece].solar_tra[1] = orig_tra[1] + transmission_decrease
                        model.pieces[piece].solar_tra[2] = orig_tra[2]  # absorption unchanged
            
            # Skip rows with missing data
            if pd.isna(air_C) or pd.isna(solar) or pd.isna(sky):
                print("Skipped, missing air/solar/sky data")
                continue

            # if pd.isna(real_caf2) or pd.isna(real_sapph) or pd.isna(real_boro):
            #     continue
            
            # Calculate elapsed time since initial time
            current_datetime = row['datetime']
            elapsed_seconds_total = (current_datetime - initial_time).total_seconds()
            elapsed_seconds_sim = models['caf2'].n_steps * models['caf2'].step__s

            secs_to_model = (elapsed_seconds_total - elapsed_seconds_sim)
            # print(initial_time, current_datetime, elapsed_seconds_total, elapsed_seconds_sim, secs_to_model)
            steps_taken_now = 0

            while steps_taken_now*models['caf2'].step__s < secs_to_model:
                for model in models.values():
                    model.step(
                        Ta__C=air_C,
                        insolation__W_m2=solar,
                        sky__W_m2=sky,
                    )
                steps_taken_now += 1

            # assert the sim time now is within 1 second of the real time
            sim_seconds = models['caf2'].n_steps * models['caf2'].step__s
            divergence_secs = ((initial_time + pd.Timedelta(seconds=sim_seconds)) - current_datetime).total_seconds()
            if abs(divergence_secs) > 1:
                raise ValueError("Simulation time diverged from real time by %.2f > 1 second" % (
                    divergence_secs
                ))

            # Store temperatures every using total simulation time, round to nearest decimal
            def rnd(f):
                return round(f, 1) if f is not None else None

            # Get unrounded temperatures for accurate difference calculations
            caf2_surf_unrounded = models['caf2'].pieces['surf'].T__C
            sapph_surf_unrounded = models['sapphire'].pieces['surf'].T__C
            boro_surf_unrounded = models['borosilicate'].pieces['surf'].T__C
            
            # Store rounded individual temperatures
            modeled_temps['datetime'].append(initial_time + pd.Timedelta(seconds=sim_seconds))
            modeled_temps['caf2_surf'].append(rnd(caf2_surf_unrounded))
            modeled_temps['caf2_pane'].append(rnd(models['caf2'].avg_pane__C()))
            modeled_temps['sapph_surf'].append(rnd(sapph_surf_unrounded))
            modeled_temps['sapph_pane'].append(rnd(models['sapphire'].avg_pane__C()))
            modeled_temps['boro_surf'].append(rnd(boro_surf_unrounded))
            modeled_temps['boro_pane'].append(rnd(models['borosilicate'].avg_pane__C()))
            
            # Compute relative temperatures from unrounded values, then round
            modeled_temps['boro_vs_caf2'].append(rnd(boro_surf_unrounded - caf2_surf_unrounded))
            modeled_temps['sapph_vs_caf2'].append(rnd(sapph_surf_unrounded - caf2_surf_unrounded))

            # Get modeled temperatures to output
            mod_caf2_surf = models['caf2'].pieces['surf'].T__C
            mod_sapph_surf = models['sapphire'].pieces['surf'].T__C
            mod_boro_surf = models['borosilicate'].pieces['surf'].T__C
            
            # Check if temperature rankings match between real and modeled data
            # Get the rankings (1=highest, 2=middle, 3=lowest)
            real_temps = {'caf2': real_caf2, 'sapph': real_sapph, 'boro': real_boro}
            model_temps = {'caf2': mod_caf2_surf, 'sapph': mod_sapph_surf, 'boro': mod_boro_surf}
            
            # Sort by temperature to get ranking
            real_sorted = sorted(real_temps.items(), key=lambda x: x[1], reverse=True)
            model_sorted = sorted(model_temps.items(), key=lambda x: x[1], reverse=True)
            
            # Extract just the material order
            real_order = [mat for mat, temp in real_sorted]
            model_order = [mat for mat, temp in model_sorted]
            
            # Check if orders match
            ranking_match = real_order == model_order
            modeled_temps['ranking_match'].append(ranking_match)

            # Calculate deltas (modeled - real)
            delta_caf2 = mod_caf2_surf - real_caf2
            delta_sapph = mod_sapph_surf - real_sapph
            delta_boro = mod_boro_surf - real_boro
            
            # Format and print output
            time_str = row['datetime'].strftime('%H:%M:%S')
            print(f"{time_str:13} {air_C:4.1f} {solar:6.1f} {sky:5.1f} |       "
                  f"{real_caf2:4.1f} {real_sapph:5.1f} {real_boro:4.1f} |        "
                  f"{mod_caf2_surf:4.1f} "
                  f"{mod_sapph_surf:5.1f} "
                  f"{mod_boro_surf:4.1f} |        "
                  f"{delta_caf2:+5.2f} {delta_sapph:+6.2f} {delta_boro:+5.2f}")
        
        print()
        print("Trace simulation complete.")
        
        # Convert modeled temps to DataFrame for plotting
        modeled_df = pd.DataFrame(modeled_temps)
        
        # Generate plot if requested
        if plot:
            print("Generating plot...")
            plot_reality_trace(df, modeled_df, infilled_indices=infilled_indices if infill_missing_data else None)


if __name__ == '__main__':
    fire.Fire(CmdLine)
