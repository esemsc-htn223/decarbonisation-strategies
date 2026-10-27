__all__ = ['Term', 'TermCollection', 'TermGroup', 'TERMS_ALL', 'TERMS_SUMMARY', 'TERMS_SIMPLE']

import os
import matplotlib.pyplot as plt
from typing import Iterable
from dataclasses import dataclass
import uuid
import json
from typing import Literal, get_args

from decarb.util import TERMS_DIR, OUT_DIR, logger

TermGroup = Literal['fossil', 'decarbonisation']

@dataclass
class Term:
    alias: str
    regex: str
    colour: str
    prompt_alias: str|None = None
    group: TermGroup = 'decarbonisation'

    #linestyle: str|None = None
    dash: bool = False
    marker: str|None = None
    hatch: str|None = None
    edgecolour: str|None = None
    dash_ls = (0, (5, 10)) # custom dashed linestyle for terms with dash=True
    variants: dict|None = None
    variants_iea_indexer : dict|None = None  # optional indexers for the IEA ETP clean energy guide .csv (https://www.iea.org/data-and-statistics/data-tools/etp-clean-energy-technology-guide)

    def __post_init__(self):
        if self.prompt_alias is None:
            self.prompt_alias = self.alias
        if self.group not in get_args(TermGroup):
            raise ValueError(f'Term {self.alias}: group must be one of {get_args(TermGroup)}. Got {self.group!r}')
        self.prompt_fname = self.alias.lower().replace(' ', '-').replace('(', '').replace(')', '') + '.txt'

    def __str__(self):
        return self.alias

    def __repr__(self):
        return f'Term(alias={self.alias}, regex={self.regex}, colour={self.colour}, prompt_alias={self.prompt_alias}, group={self.group})'
     
    def __hash__(self):
        return hash(self.alias)
     
    def __eq__(self, other):
        if isinstance(other, Term):
            return (self.alias == other.alias and 
                    self.regex == other.regex)
        return False
     
    def __ne__(self, other):
        return not self.__eq__(other)
     
    def style_eq(self, other):
        if isinstance(other, Term):
            return (self.colour == other.colour and 
                    self.prompt_alias == other.prompt_alias and
                    self.dash == other.dash and
                    self.marker == other.marker and
                    self.hatch == other.hatch and
                    self.edgecolour == other.edgecolour)
        return False

    def copy(self):
        return Term(
            alias=self.alias,
            regex=self.regex,
            colour=self.colour,
            prompt_alias=self.prompt_alias,
            group=self.group,
            dash=self.dash,
            marker=self.marker,
            hatch=self.hatch,
            edgecolour=self.edgecolour
        )
     
    def set_collection(self, collection: 'TermCollection'):
        if hasattr(self, 'collection'):
            raise AttributeError(f'Term {self.alias} is already part of a collection. Cannot set collection again.')
        self.collection = collection
        return self  # allow chaining
    

class TermCollection:
    def __init__(self, terms: Iterable[Term], folder_name: str):

        # check that all aliases are unique
        aliases = [term.alias for term in terms]
        if len(aliases) != len(set(aliases)):
            raise ValueError('All term aliases must be unique')

        for i, term in enumerate(terms):
            try:
                term.set_collection(self)
            except AttributeError:
                terms[i] = term.copy().set_collection(self)

        self.terms = sorted(list(terms), key = lambda t: t.alias.lower())
        self.aliases = [term.alias for term in self.terms]
        self.regexes = [term.regex for term in self.terms]
        self.folder_name = folder_name.replace(' ', '-').lower()
        self.colours = [term.colour for term in self.terms]
        self.groups = {term.group for term in self.terms}

        self.folder = os.path.join(TERMS_DIR, self.folder_name)
        self.json_path = os.path.join(self.folder, 'terms.json')
        self.batches_folder = os.path.join(self.folder, 'batches')
        self.prompts_folder = os.path.join(self.folder, 'prompts')

        self.out_folder = os.path.join(OUT_DIR, self.folder_name)

        # set up folder structure
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)
        terms_data = [{
            'alias': term.alias,
            'prompt_fname': term.prompt_fname,
            'regex': term.regex,
            'colour': term.colour,
            'prompt_alias': term.prompt_alias,
            'group': term.group} for term in self.terms]

        # determine if changes have been made to the terms, and if so, rewrite the json file and warn the user
        existing = None
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r') as f:
                existing = json.load(f)
        if existing != terms_data:
            if existing is not None:
                logger.warning(
                    f'TERMS CHANGED: "{self.folder_name}" definition differs from {self.json_path}. '
                    'Rewriting. Any search results, prompts or batches already stored under this '
                    'folder were produced with the previous terms.'
                )
            with open(self.json_path, 'w') as f:
                json.dump(terms_data, f, indent=4)
        if not os.path.exists(self.batches_folder):
            os.makedirs(self.batches_folder)
        if not os.path.exists(self.prompts_folder):
            os.makedirs(self.prompts_folder)

        self.uuid = 'terms-' + uuid.uuid5(uuid.NAMESPACE_OID, ''.join(sorted([term.alias + term.regex for term in self.terms]))).hex
        with open(os.path.join(self.folder, 'uuid.txt'), 'w') as f:
            f.write(self.uuid)

        logger.debug(f'CREATED TermCollection: folder_name={self.folder_name}, uuid={self.uuid}, terms={[term.alias for term in self.terms]}')

    @classmethod
    def from_folder(cls, folder_name: str):
        if os.sep in folder_name:
            fname = os.path.basename(folder_name)
        else:
            fname = folder_name.replace(' ', '-').lower()
        json_path = os.path.join(TERMS_DIR, fname, 'terms.json')
        if not os.path.exists(json_path):
            raise ValueError(f'TermCollection with folder name {fname} does not exist')
        with open(json_path, 'r') as f:
            terms_data = json.load(f)
        terms = [Term(alias=term_data['alias'], regex=term_data['regex'], colour=term_data['colour'], prompt_alias=term_data.get('prompt_alias'), group=term_data.get('group', 'decarbonisation')) for term_data in terms_data]
        return cls(terms, fname)

    @classmethod
    def from_uuid(cls, uuid_str: str):
        for folder_name in os.listdir(TERMS_DIR):
            uuid_path = os.path.join(TERMS_DIR, folder_name, 'uuid.txt')
            if os.path.exists(uuid_path):
                with open(uuid_path, 'r') as f:
                    if f.read().strip() == uuid_str:
                        return cls.from_folder(folder_name)
        raise ValueError(f'TermCollection with uuid {uuid_str} does not exist')

    def by_group(self, group: TermGroup):
        return TermSubCollection([term for term in self.terms if term.group == group], self)

    def __iter__(self):
        return iter(self.terms)
    
    def __getitem__(self, key: str|int):
        '''Index by alias, lowecase index, regex, or, prompt_fname. Priority in that order '''

        def check_term(term: Term, key: str):
            # alias indexing
            if term.alias == key:
                return True
            if term.alias.lower() == key.lower():
                return True
               
            # regex
            if term.regex == key:
                return True
               
            # prompt fname indexing
            if term.prompt_fname == key:
                return True
            if term.prompt_fname.replace('.txt', '') == key:
                return True

            return False

        if isinstance(key, int):
            return self.terms[key]
        elif isinstance(key, slice):
            return TermSubCollection(self.terms[key], self)
        elif isinstance(key, str):
            try:
                return next(term for term in self.terms if check_term(term, key))
            except StopIteration:
                raise KeyError(f'Term with alias, prompt_fname {key} not found')
        elif isinstance(key, Term):
            if key in self.terms:
                return key
            else:
                raise KeyError(f'Term {key.alias} not found in collection')
        elif isinstance(key, Iterable):
            return TermSubCollection([self[k] for k in key], self)
        else:
            raise TypeError(f'Invalid key type: {type(key)}. Must be str, int, slice, Term, or Iterable of those types.')
     
    def __contains__(self, key: str|int|Term):
        if isinstance(key, Term):
            return key in self.terms
        try:
            self.__getitem__(key)
            return True
        except KeyError:
            return False
    
    def __len__(self):
        return len(self.terms)

    def __repr__(self):
        return f'TermCollection(folder_name={self.folder_name}, terms={[term.alias for term in self.terms]})'
     
    def __str__(self):
        return self.folder_name

    def plot_colours(self):
        _fig, (_ax1, _ax2) = plt.subplots(nrows=2, figsize = (5, 6))
        for term in self.terms:
            _ax1.plot([], [], label=' ', color=term.colour, linestyle=term.dash_ls if term.dash else None, marker = None, gapcolor=term.edgecolour)
            _ax1.scatter([], [], color=term.colour, marker = term.marker, hatch=term.hatch, edgecolor=term.edgecolour, label=' ')
        _ax1.legend(loc='center', ncols=len(self.terms), frameon=False, markerscale=1)
        _ax1.axis('off')

        # pichart of all colours and hatches
        _wedges, _ = _ax2.pie([1]*len(self.terms), labels=[term.alias for term in self.terms], colors=[term.colour for term in self.terms], hatch=[term.hatch for term in self.terms], rotatelabels=True)#, edgecolor=[term.edgecolour for term in self.terms], startangle=90)
        for wedge, term in zip(_wedges, self.terms):
            if term.edgecolour:
                wedge.set_edgecolor(term.edgecolour)

        return _fig
     
class TermSubCollection(TermCollection):
    def __init__(self, terms: Iterable[Term], parent_collection: TermCollection):
        self.parent_collection = parent_collection
        self.terms = sorted(list(terms), key = lambda t: t.alias.lower())

        self.aliases = [term.alias for term in self.terms]
        self.regexes = [term.regex for term in self.terms]
        self.folder_name = parent_collection.folder_name
        self.colours = [term.colour for term in self.terms]
        self.groups = {term.group for term in self.terms}

        self.folder = parent_collection.folder
        self.json_path = parent_collection.json_path
        self.batches_folder = parent_collection.batches_folder
        self.prompts_folder = parent_collection.prompts_folder
        self.out_folder = parent_collection.out_folder

        self.uuid = parent_collection.uuid + '-sub-' + uuid.uuid5(uuid.NAMESPACE_OID, ''.join([term.alias + term.regex for term in self.terms])).hex

    @classmethod
    def from_parent(cls, parent_collection: TermCollection, keys: Iterable):
        terms = [parent_collection[k] for k in keys]
        return cls(terms, parent_collection)

    @classmethod
    def from_folder(cls, folder_name: str):
        raise NotImplementedError('TermSubCollection cannot be created from folder. Use TermCollection.from_folder() to create a parent collection first, then use TermSubCollection.from_parent() to create a subcollection.')

    def __repr__(self):
        return f'TermSubCollection(parent_collection={self.parent_collection.folder_name}, terms={[term.alias for term in self.terms]})'
    

# ----------------------------- REGEX ----------------------------- #
SOLAR_REGEX = r'''\b(solar\s+(power|energy|farm|park|plant|panel|cell|module|PV|photovoltaic|thermal|roof|rooftop|capacity|generation|project|investment|asset|portfolio|array|system|technology)|
photovoltaic(?!s?\s+effect)|
\bPV\b(?=\s+(power|energy|farm|park|plant|panel|capacity|generation|project|asset|portfolio|array|system))|
concentrated\s+solar|
CSP\b|
floating\s+solar|
rooftop\s+solar|
utility.scale\s+solar|
solar\s+\w+\s+(GW|MW|TWh|GWh))'''.replace('\n', '')

WIND_REGEX = r'''\b(wind\s+(power|energy|farm|park|plant|turbine|capacity|generation|project|investment|asset|portfolio|offshore|onshore|system|technology)|
offshore\s+wind|
onshore\s+wind|
floating\s+wind|
wind\s+\w+\s+(GW|MW|TWh|GWh)|
\bWTG\b|
wind\s+turbine)'''.replace('\n', '')

GREEN_HYDROGEN_REGEX = r'''\b(green\s+hydrogen|
renewable\s+hydrogen|
electrolytic\s+hydrogen|
hydrogen\s+electrolys[ie]s|
electrolys[ie]s.based\s+hydrogen|
PEM\s+(electroly[sz]er|hydrogen)|
alkaline\s+(electroly[sz]er|hydrogen)|
electroly[sz]er(?!\s+(for\s+)?chlor)|
Power.to.(X|hydrogen|gas|H2)|
P2(X|H|G)\b|
e.hydrogen|
\bH2\b(?=\s+(from|via|using|produced\s+by)\s+(renewable|electrolysis|wind|solar|clean|green)))'''.replace('\n', '')

BLUE_HYDROGEN_REGEX = r'''\b(blue\s+hydrogen|
low.carbon\s+hydrogen(?!\s+from\s+(electrolysis|renewable))|
SMR.CCS|
ATR.CCS|
autothermal\s+reform\w+\s+with\s+(carbon\s+capture|CCS)|
steam\s+methane\s+reform\w+\s+with\s+(carbon\s+capture|CCS)|
hydrogen\s+(from|via)\s+(natural\s+gas|methane|fossil)\s+with\s+(CCS|carbon\s+capture))'''.replace('\n', '')

HYDROGEN_REGEX = r'''\b(hydrogen\s+(fuel|economy|production|storage|transport|distribution|infrastructure|hub|valley|strategy|roadmap|investment|project|asset|technology|demand|supply|market|network|pipeline|carrier|energy|power|mobility|vehicle|bus|truck|ship|aviation|heating|industry)|
hydrogen\s+fuel\s+cell|
fuel\s+cell\s+(vehicle|technology|hydrogen)|
\bFCEV\b|
hydrogen.powered|
low.carbon\s+hydrogen|
clean\s+hydrogen|
decarbonised?\s+hydrogen|
hydrogen\s+\w+\s+(GW|MW|GWh|kt|Mt|tonne))'''.replace('\n', '')

CCS_REGEX = r'''\b(carbon\s+capture(\s+and\s+(storage|sequestration|utili[sz]ation|utilisation))?|
CCS\b|
CCUS\b|
BECCS\b|
carbon\s+sequestration|
CO2\s+(capture|storage|sequestration|injection|storage|utilisation|utilization)|
geological\s+(storage|sequestration)\s+of\s+(carbon|CO2)|
enhanced\s+oil\s+recovery(?=.*CO2)|
post.combustion\s+capture|
pre.combustion\s+capture|
oxy.fuel\s+capture|
direct\s+air\s+capture|
\bDAC\b|
point.source\s+capture|
carbon\s+storage\s+(hub|cluster|facility|site|reservoir)|
saline\s+(aquifer|formation)\s+(storage|injection))'''.replace('\n', '')

BIOFUEL_REGEX = r'''\b(biofuel|
biodiesel|
bioethanol|
bio.jet(\s+fuel)?|
sustainable\s+aviation\s+fuel|
\bSAF\b|
hydrotreated\s+vegetable\s+oil|
\bHVO\b|
renewable\s+diesel|
biomass(\s+(energy|power|fuel|feedstock|combustion|gasification|co.firing|plant))?|
biomass.to.(liquid|gas|power|energy)|
\bBtL\b|
\bBtG\b|
biogas(?!\s*(upgrading|network|grid|injection))| 
lignocellulosic|
advanced\s+biofuel|
second.generation\s+(biofuel|ethanol)|
cellulosic\s+(ethanol|biofuel)|
bio.based\s+fuel\b)'''.replace('\n', '')

BIOGAS_REGEX = r'''\b(biomethane|
renewable\s+natural\s+gas|
\bRNG\b|
biogas\s+(upgrading|purification|injection|network|grid|production|plant|facility|project)|
anaerobic\s+digestion|
\bAD\b(?=\s+(plant|facility|project|gas|capacity))|
landfill\s+gas|
bio.SNG|
synthetic\s+natural\s+gas\s+(from\s+bio|bio.based))'''.replace('\n', '')

GEOTHERMAL_REGEX = r'''\b(geothermal(\s+(energy|power|heat|heating|plant|capacity|generation|project|investment|resource|well|reservoir|system))?|
ground.source\s+heat\s+pump|
\bGSHP\b|
hot\s+dry\s+rock|
enhanced\s+geothermal\s+system|
\bEGS\b)'''.replace('\n', '')

TIDAL_REGEX = r'''\b(tidal(\s+(energy|power|stream|barrage|lagoon|turbine|project|farm))?|
wave\s+(energy|power|converter|device|project|farm)|
ocean\s+(thermal|energy\s+conversion|current\s+energy)|
\bOTEC\b|
marine\s+(energy|power|current|renewable)|
tidal\s+stream|
tidal\s+range)'''.replace('\n', '')

STORAGE_REGEX = r'''\b(battery\s+(storage|energy\s+storage|technology|system|cell|pack|chemistry|project|investment|capacity|gigafactory)|
\bBESS\b|
lithium.ion\s+(battery|storage)|
grid.scale\s+(battery|storage)|
utility.scale\s+(battery|storage)|
pumped.hydro(\s+(storage|power))?|
compressed\s+air\s+energy\s+storage|
\bCAES\b|
flow\s+battery|
vanadium\s+(redox|flow)\s+battery|
thermal\s+energy\s+storage|
long.duration\s+(energy\s+)?storage|
\bLDES\b|
solid.state\s+battery|
energy\s+storage\s+(system|project|capacity|investment|asset|technology))'''.replace('\n', '')

EVS_REGEX = r'''\b(electric\s+vehicle|
\bEV\b(?=\s+(charging|fleet|infrastructure|market|adoption|transition|sales|demand))|
battery\s+electric\s+vehicle|
\bBEV\b|
plug.in\s+(hybrid|vehicle)|
\bPHEV\b|
\bFCEV\b|
EV\s+charg(ing|er|point)|
charging\s+(infrastructure|network|station|point|hub|post)|
ultra.rapid\s+charg|
fast.charg(?:ing|er)|
charge\s+point\s+operator|
\bCPO\b(?=\s+(network|business|partnership))|
vehicle.to.grid|
\bV2G\b)'''.replace('\n', '')

NATURE_REGEX = r'''\b(nature.based\s+solution|
\bNBS\b(?=\s+(project|offset|credit|investment|programme|carbon))|
natural\s+climate\s+solution|
\bNCS\b(?=\s+(project|offset|credit|carbon))|
carbon\s+offset|
carbon\s+credit|
voluntary\s+carbon\s+market|
\bVCM\b|
REDD\+?|
afforestation|
reforestation|
avoided\s+deforestation|
blue\s+carbon|
soil\s+carbon\s+(sequestration|offset|credit)|
forest\s+(carbon|offset|credit|protection|conservation)|
mangrove\s+(restoration|carbon|project)|
peatland\s+(restoration|carbon|protection))'''.replace('\n', '')

EFFICIENCY_REGEX = r'''\b(energy\s+efficienc(y|ies)(?!\s+(standard|label|rating|of\s+appliance))|
energy\s+intensit(y|ies)\s+(reduction|improvement|target|performance)|
heat\s+recovery|
waste\s+heat\s+(recovery|utilisation|reuse)|
co.generation|
cogeneration|
combined\s+heat\s+and\s+power|
\bCHP\b|
demand.side\s+(management|response|reduction)|
\bDSM\b(?=\s+(programme|measure|target))|
electrification\s+of\s+(heat|process|industry|operations)|
industrial\s+electrification|
methane\s+(reduction|abatement|mitigation|intensity|flaring|venting|leak))'''.replace('\n', '')

NUCLEAR_CONVENTIONAL_REGEX = r'''\b(nuclear\s+(power|energy|plant|station|reactor|capacity|generation|investment|asset|fleet|fuel|fission|programme|project|technology|electricity)|
nuclear\s+power\s+(plant|station|unit|block)|
\bNPP\b|
pressurised\s+water\s+reactor|
boiling\s+water\s+reactor|
\bPWR\b|
\bBWR\b|
light\s+water\s+reactor|
\bLWR\b|
CANDU\s+reactor|
nuclear\s+fuel\s+(cycle|assembly|rod|enrichment|reprocessing)|
uranium\s+(fuel|enrichment|supply|procurement)|
spent\s+nuclear\s+fuel|
nuclear\s+waste\s+(storage|disposal|management)|
deep\s+geological\s+(disposal|repository)\s+(of|for)\s+nuclear|
nuclear\s+decommissioning)'''.replace('\n', '')

NUCLEAR_SMR_REGEX = r'''\b(small\s+modular\s+reactor|
\bSMR\b(?!\s*(emissions|scope|reduction|target|intensity|performance))| 
advanced\s+modular\s+reactor|
\bAMR\b(?=\s+(technology|reactor|design|deployment|project|programme|investment|capacity|fleet))|
micro\s+(nuclear\s+)?(reactor|SMR)|
micro\s+modular\s+reactor|
\bMMR\b(?=\s+(technology|reactor|design|deployment|project))|
factory.built\s+(reactor|nuclear)|
modular\s+(nuclear\s+)?(reactor|design|technology|deployment)|
advanced\s+reactor\s+(design|concept|technology|deployment|programme)|
generation\s+IV\s+(reactor|nuclear|technology)|
\bGen\s*IV\b|
molten\s+salt\s+reactor|
\bMSR\b(?=\s+(technology|reactor|design|deployment|project))|
fast\s+(neutron\s+)?reactor|
sodium.cooled\s+(fast\s+)?reactor|
\bSFR\b(?=\s+(technology|reactor|design|deployment|project))|
gas.cooled\s+(fast\s+)?reactor|
high.temperature\s+(gas.cooled\s+)?reactor|
\bHTGR\b|
\bHTR\b(?=\s+(technology|reactor|design|deployment|project))|
pebble.bed\s+reactor|
integral\s+pressurised\s+water\s+reactor|
\biPWR\b|
nuclear\s+microreactor|
nuclear\s+(heat|cogeneration)\s+(for\s+)?(industrial|district|process)|
nuclear\s+hydrogen\s+(production|co.generation))'''.replace('\n', '')

NUCLEAR_FUSION_REGEX = r'''\b(nuclear\s+fusion|
fusion\s+(energy|power|reactor|plant|technology|research|investment|programme|project|venture|start.?up|company|partner\w+)|
thermonuclear\s+fusion|
inertial\s+confinement\s+fusion|
\bICF\b(?=\s+(energy|fusion|reactor|project|technology))|
magnetic\s+confinement\s+fusion|
\bMCF\b(?=\s+(energy|fusion|reactor|project|technology))|
tokamak|
stellarator|
\bITER\b|
\bDEMO\b(?=\s+(reactor|fusion|plant|project|programme))|
deuterium.tritium\s+(fuel|reaction|fusion)|
\bD.T\s+fusion\b|
plasma\s+(confinement|heating|physics)\s+(for|in)\s+fusion|
fusion\s+ignition|
net\s+energy\s+(gain|positive)\s+(from\s+)?fusion|
inertial\s+fusion\s+energy|
\bIFE\b(?=\s+(project|technology|company|investment))|
compact\s+fusion\s+(reactor|device|technology)|
private\s+fusion\s+(company|venture|investment|start.?up)|
fusion\s+pilot\s+plant|
\bFPP\b(?=\s+(fusion|project|programme|investment)))'''.replace('\n', '')

# ----------------------------- VARIANTS ----------------------------- #


TECHNOLOGY_VARIANTS = {
    'Solar': {
        'utility_scale_pv': "ground-mounted solar farms supplying the grid or large offtakers",
        'distributed_pv': "rooftop or on-site solar for homes, businesses or company facilities",
        'concentrated_solar': "concentrated solar power (CSP) or solar thermal",
        'solar_manufacturing': "making solar cells, wafers or panels",
    },
    'Wind': {
        'onshore': "land-based wind",
        'offshore': "offshore wind (fixed-bottom, or type unstated)",
        'floating': "floating offshore wind (also tag 'offshore')",
    },
    'Hydrogen': {
        'green': "electrolytic hydrogen from renewable power",
        'blue': "hydrogen from natural gas or other fossil fuels with carbon capture",
        'grey': "hydrogen from fossil fuels without carbon capture",
        'low_carbon_unspecified': "described as low-carbon or clean without stating the production route",
        'mobility': "fuel-cell vehicles and hydrogen refuelling stations",
        'derivatives': "ammonia, methanol, e-fuels or other products made from hydrogen",
    },
    'CCS': {
        'industrial_capture': "capture from industrial sources: refineries, LNG, gas processing, chemicals, cement, steel",
        'power_capture': "capture from power generation",
        'direct_air_capture': "direct air capture (DAC)",
        'beccs': "bioenergy with carbon capture",
        'transport_and_storage': "CO2 pipelines, shipping, storage sites or hubs, including storage offered as a service",
        'enhanced_oil_recovery': "CO2 injected to increase oil production",
        'utilisation': "CO2 used to make products",
    },
    'Biofuel': {
        'ethanol': "bioethanol",
        'renewable_diesel': "biodiesel, HVO or renewable diesel",
        'sustainable_aviation_fuel': "SAF",
        'biogas': "biogas, biomethane or renewable natural gas (RNG)",
        'biomass_power_heat': "biomass burned for power or heat",
        'advanced_feedstocks': "algae, cellulosic, waste or residue feedstocks",
    },
    'Geothermal': {
        'power': "geothermal electricity generation",
        'direct_heat': "geothermal heat for buildings or industry",
        'ground_source_heat_pumps': "ground-source heat pumps",
        'advanced_geothermal': "enhanced or closed-loop geothermal (EGS)",
    },
    'Tidal': {
        'tidal_stream': "tidal stream turbines",
        'tidal_range': "tidal barrages or lagoons",
        'wave': "wave energy",
        'ocean_thermal': "ocean thermal energy conversion (OTEC)",
    },
    'Storage': {
        'grid_scale_battery': "utility or grid-scale battery storage (BESS)",
        'behind_the_meter_battery': "battery storage at homes, businesses, sites or charging points",
        'battery_technology': "battery chemistry, cells or manufacturing (e.g. solid-state, sodium-ion)",
        'pumped_hydro': "pumped hydro storage",
        'compressed_air': "compressed air energy storage (CAES)",
        'other_long_duration': "flow batteries, thermal or other long-duration storage",
    },
    'EVs': {
        'public_charging': "public charge points, hubs or networks",
        'fleet_charging': "depot or fleet charging for businesses",
        'home_charging': "residential charging",
        'vehicles': "electric vehicles themselves, including the company's own fleet",
        'batteries_and_materials': "EV batteries or battery materials such as lithium",
        'ev_fluids': "lubricants, coolants or other fluids for EVs",
        'vehicle_to_grid': "vehicle-to-grid (V2G)",
    },
    'Nature': {
        'reforestation': "afforestation or reforestation",
        'avoided_deforestation': "forest protection, REDD+",
        'blue_carbon': "mangroves, seagrass or other coastal ecosystems",
        'agriculture_and_soil': "soil carbon or regenerative agriculture",
        'credit_trading': "originating, trading or retiring carbon credits",
        'offset_products': "products sold as carbon neutral or offset (e.g. offset fuels or LNG)",
    },
    'Nuclear': {
        'conventional_reactors': "large-scale fission reactors",
        'small_modular_or_advanced': "small modular or advanced reactors",
        'nuclear_fuel_cycle': "uranium mining, enrichment or fuel",
        'power_offtake': "buying nuclear-generated power",
    },
    'Fusion': {
        'magnetic_confinement': "tokamaks, stellarators or other magnetic confinement",
        'inertial_confinement': "laser or other inertial confinement",
        'alternative_approach': "other or unstated fusion approaches (e.g. field-reversed configuration)",
    },
    'Coal': {
        'thermal_coal': "coal mined for power generation",
        'metallurgical_coal': "coking or metallurgical coal",
        'coal_power': "coal-fired power generation",
        'coal_conversion': "coal-to-liquids or coal gasification",
    },
    'Oil': {
        'onshore': "conventional onshore oil",
        'shallow_water': "shallow-water offshore oil",
        'deepwater': "deepwater or ultra-deepwater oil",
        'arctic': "Arctic or other frontier oil",
        'enhanced_recovery': "enhanced or improved oil recovery",
    },
    'Natural gas': {
        'upstream_production': "gas exploration and production",
        'lng': "liquefied natural gas",
        'pipeline': "pipeline gas transport and supply",
        'gas_power': "gas-fired power generation",
        'transport_fuel': "CNG or LNG as a vehicle or marine fuel",
        'ngls': "natural gas liquids",
    },
    'Refining': {
        'fuels_refining': "conventional refining of fuels",
        'lubricants': "lubricant base oils and blending",
        'biorefinery_conversion': "converting refineries to process bio-feedstocks",
        'co_processing': "processing bio-feedstocks alongside crude",
        'integrated_chemicals': "refineries integrated with chemical plants",
    },
    'Petrochemicals': {
        'olefins': "ethylene, propylene and other olefins",
        'aromatics': "benzene, paraxylene and other aromatics",
        'polymers': "polyethylene, polypropylene and other plastics",
        'specialty_chemicals': "specialty or performance chemicals",
        'chemical_recycling': "advanced or chemical recycling of plastics",
    },
    'Unconventional': {
        'tight_oil': "shale or tight oil",
        'tight_gas': "shale or tight gas",
        'oil_sands': "oil sands, bitumen and synthetic crude",
        'heavy_oil': "heavy oil",
        'coalbed_methane': "coalbed methane",
    },
}

TECHNOLOGY_VARIANTS_IEA: dict[str, dict[str, str]] = {
    'Solar': {
        'photovoltaic': 'solar PV, crystalline silicon or cell type not stated',
        'thin_film_pv': 'thin-film PV: perovskite, CIGS, CdTe, organic or amorphous/microcrystalline silicon',
        'floating_pv': 'floating solar PV',
        'advanced_pv': 'concentrated PV, multi-junction cells or space solar',
        'concentrated_solar_power': 'solar thermal electricity (CSP): parabolic trough, tower or Fresnel',
        'solar_thermal_heat': 'solar thermal for heat or district heating',
        'manufacturing': 'making solar cells, wafers or panels (not an IEA technology)',
    },
    'Wind': {
        'onshore': 'land-based wind',
        'offshore': 'offshore wind with foundation type not stated',
        'offshore_fixed': 'offshore wind explicitly on fixed foundations (monopile, jacket, gravity base)',
        'offshore_floating': 'floating offshore wind',
        'airborne': 'airborne wind energy (kites, tethered systems)',
        'offshore_hydrogen': 'hydrogen produced offshore from wind power',
    },
    'Hydrogen': {
        'electrolysis': "hydrogen from electrolysis of water, including 'green' hydrogen",
        'fossil_with_ccus': "hydrogen from natural gas or coal with carbon capture, including 'blue' hydrogen",
        'methane_pyrolysis': "hydrogen from splitting methane into hydrogen and solid carbon, including 'turquoise' hydrogen",
        'bio_hydrogen': 'hydrogen from biomass or waste gasification, or biological production',
        'natural_hydrogen': "naturally occurring (geologic or 'white') hydrogen",
        'unabated_fossil': "hydrogen from fossil fuels without carbon capture, including 'grey' hydrogen (not an IEA clean technology)",
        'low_carbon_unspecified': 'described as low-carbon, clean or renewable without stating the production route',
        'infrastructure': 'hydrogen pipelines, storage, liquefaction, shipping or carriers',
        'power_generation': 'hydrogen or ammonia used in turbines, fuel cells or co-firing for power',
        'hydrogen_based_fuels': 'ammonia, methanol, e-fuels or other synthetic fuels made from hydrogen',
        'transport_use': 'fuel-cell or hydrogen vehicles, ships or aircraft, and refuelling or bunkering',
        'industrial_use': 'hydrogen used in refining, chemicals, steel or other industry',
    },
    'CCS': {
        'direct_air_capture': 'direct air capture (DAC)',
        'power_capture': 'capture from coal- or gas-fired power generation',
        'industrial_capture': 'capture from refineries, gas processing, LNG, hydrogen production, chemicals, cement, steel or other industry',
        'bioenergy_capture': 'capture from bioenergy (BECCS) or biofuel production',
        'transport': 'CO2 pipelines, shipping or terminals',
        'geological_storage': 'storage in saline aquifers or depleted oil and gas fields, and storage monitoring',
        'mineral_storage': 'storage by mineralisation',
        'enhanced_oil_recovery': 'CO2 injected to increase oil production',
        'utilisation': 'CO2 used to make fuels, chemicals or materials',
    },
    'Biofuel': {
        'conventional_ethanol': 'ethanol from sugar or starch crops (e.g. sugarcane, corn)',
        'cellulosic_ethanol': 'ethanol from lignocellulosic biomass: straw, wood, residues',
        'biodiesel': 'biodiesel (FAME) from transesterification of oils and fats',
        'hvo_hefa': 'renewable diesel or jet fuel from hydrotreating oils and fats (HVO, HEFA)',
        'alcohol_to_jet': 'jet fuel from ethanol, butanol or methanol',
        'biogas_biomethane': 'biogas, biomethane or renewable natural gas, from anaerobic digestion or gasification',
        'gasification_to_liquids': 'liquid fuels from biomass gasification and Fischer-Tropsch synthesis',
        'pyrolysis_and_liquefaction': 'bio-oils from pyrolysis or hydrothermal liquefaction',
        'biomass_power_heat': 'biomass burned for power or heat',
        'sustainable_aviation_fuel': 'SAF, whatever the route; also tag the route if stated (not an IEA technology)',
        'algae': 'algae-based biofuels (not in the IEA guide)',
    },
    'Geothermal': {
        'conventional_power': 'geothermal power from dry steam, flash or binary (ORC, Kalina) plants',
        'next_generation': 'enhanced (EGS), closed-loop or superhot geothermal',
        'drilling': 'geothermal drilling technology',
        'ground_source_heat_pumps': 'ground-source heat pumps',
        'direct_heat': 'geothermal heat used directly for buildings, district heating or industry (not in the IEA guide)',
    },
    'Tidal': {
        'tidal_stream': 'tidal stream or ocean current turbines',
        'tidal_range': 'tidal barrages or lagoons',
        'wave': 'wave energy converters',
        'ocean_thermal': 'ocean thermal energy conversion (OTEC)',
        'salinity_gradient': "salinity gradient ('blue') energy",
    },
    'Storage': {
        'lithium_ion': 'lithium-ion batteries, or battery storage with chemistry not stated',
        'sodium_ion': 'sodium-ion batteries',
        'flow_battery': 'redox flow batteries (e.g. vanadium)',
        'other_battery': 'solid-state, metal-air, zinc or other battery chemistries',
        'pumped_storage': 'pumped hydro storage',
        'compressed_air': 'compressed air energy storage (CAES)',
        'other_mechanical': 'flywheels, gravity, liquid air or liquid CO2 storage',
        'thermal_storage': 'molten salt, solid, latent-heat or other thermal energy storage',
        'battery_recycling': 'battery reuse or recycling',
    },
    'EVs': {
        'charging': 'EV charging with speed not stated (public, fleet or home)',
        'fast_charging': 'fast, rapid or ultra-fast charging',
        'battery_swapping': 'battery swapping',
        'smart_charging_v2g': 'smart charging, demand response or vehicle-to-grid',
        'electric_road_systems': 'electric road systems or dynamic charging',
        'passenger_vehicles': "battery-electric or plug-in passenger cars, including the company's own fleet",
        'commercial_vehicles': 'electric vans, trucks or buses',
        'ev_batteries': 'EV battery cells or packs',
        'battery_materials': 'lithium or other battery minerals: extraction or refining',
        'ev_fluids': 'lubricants, coolants or other fluids for EVs (not an IEA technology)',
    },
    'Nature': {
        'reforestation': 'afforestation or reforestation',
        'avoided_deforestation': 'forest protection, REDD+',
        'blue_carbon': 'mangroves, seagrass or other coastal ecosystems',
        'agriculture_and_soil': 'soil carbon or regenerative agriculture',
        'credit_trading': 'originating, trading or retiring carbon credits',
        'offset_products': 'products sold as carbon neutral or offset (e.g. offset fuels or LNG)',
    },
    'Nuclear': {
        'large_reactors': 'large-scale fission reactors (Generation III/III+ or IV)',
        'small_modular_reactors': 'small modular reactors (SMRs)',
        'fuel_cycle': 'uranium mining, conversion, enrichment, fuel fabrication, reprocessing or waste disposal',
    },
    'Fusion': {
        'magnetic_confinement': 'tokamaks, stellarators or other magnetic confinement',
        'inertial_confinement': 'laser or other inertial confinement',
        'alternative_approaches': 'magneto-inertial, electrostatic or other fusion approaches',
    },
    'Coal': {
        'thermal_coal': 'coal mined for power generation',
        'metallurgical_coal': 'coking or metallurgical coal',
        'coal_power': 'coal-fired power generation',
        'coal_conversion': 'coal-to-liquids or coal gasification',
        'methane_abatement': 'monitoring or reducing coal-mine methane',
    },
    'Oil': {
        'onshore': 'conventional onshore oil',
        'shallow_water': 'shallow-water offshore oil',
        'deepwater': 'deepwater or ultra-deepwater oil',
        'arctic': 'Arctic or other frontier oil',
        'enhanced_recovery': 'enhanced or improved oil recovery',
        'methane_abatement': 'monitoring or reducing methane leaks, venting or flaring',
    },
    'Natural gas': {
        'upstream_production': 'gas exploration and production',
        'lng': 'liquefied natural gas',
        'pipeline': 'pipeline gas transport and supply',
        'gas_power': 'gas-fired power generation',
        'transport_fuel': 'CNG or LNG as a vehicle or marine fuel',
        'ngls': 'natural gas liquids',
        'methane_abatement': 'monitoring or reducing methane leaks, venting or flaring',
    },
    'Refining': {
        'fuels_refining': 'conventional refining of fuels',
        'lubricants': 'lubricant base oils and blending',
        'biorefinery_conversion': 'converting refineries to process bio-feedstocks',
        'co_processing': 'processing bio-feedstocks alongside crude',
        'integrated_chemicals': 'refineries integrated with chemical plants',
    },
    'Petrochemicals': {
        'olefins': 'ethylene, propylene and other olefins',
        'aromatics': 'benzene, paraxylene and other aromatics',
        'polymers': 'polyethylene, polypropylene and other plastics',
        'specialty_chemicals': 'specialty or performance chemicals',
        'chemical_recycling': 'advanced or chemical recycling of plastics',
    },
    'Unconventional': {
        'tight_oil': 'shale or tight oil',
        'tight_gas': 'shale or tight gas',
        'oil_sands': 'oil sands, bitumen and synthetic crude',
        'heavy_oil': 'heavy oil',
        'coalbed_methane': 'coalbed methane',
    },
}

TECHNOLOGY_VARIANTS_IEA_INDS: dict[str, dict[str, dict[str, list[str]]]] = {
    'Solar': {
        'photovoltaic': {'tech.final.name': ['Crystalline silicon PV']},
        'thin_film_pv': {
            'tech.final.name': [
                'Chalcogenide thin-film',
                'Microcrystalline silicon PV cells',
                'Organic thin film',
                'Perovskite thin film',
            ],
        },
        'floating_pv': {'tech.final.name': ['Floating solar PV']},
        'advanced_pv': {
            'tech.final.name': ['Concentrated PV', 'Multi-junction cell PV', 'Space solar power'],
        },
        'concentrated_solar_power': {
            'tech.final.name': [
                'Linear fresnel reflector (solar)',
                'Parabolic trough (solar)',
                'Solar tower',
            ],
        },
        'solar_thermal_heat': {'tech.final.name': ['Solar thermal district heating']},
        'manufacturing': {'tech.final.name': []},
    },
    'Wind': {
        'onshore': {'tech.final.name': ['Onshore wind']},
        'offshore': {
            'tech.final.name': ['Floating offshore wind turbine', 'Seabed fixed offshore wind turbine'],
        },
        'offshore_fixed': {'tech.final.name': ['Seabed fixed offshore wind turbine']},
        'offshore_floating': {'tech.final.name': ['Floating offshore wind turbine']},
        'airborne': {'tech.final.name': ['Airborne wind energy system']},
        'offshore_hydrogen': {'tech.final.name': ['Offshore renewable hydrogen production']},
    },
    'Hydrogen': {
        'electrolysis': {
            'category.1': ['Hydrogen'],
            'category.2': ['Production'],
            'category.3': ['Electrolysis'],
        },
        'fossil_with_ccus': {
            'category.1': ['Hydrogen'],
            'category.2': ['Production'],
            'category.3': ['Thermal processes'],
            'category.4': ['CCUS-enabled'],
        },
        'methane_pyrolysis': {
            'category.1': ['Hydrogen'],
            'category.2': ['Production'],
            'category.3': ['Thermal processes'],
            'category.4': ['Pyrolysis'],
        },
        'bio_hydrogen': {
            'tech.final.name': [
                'Biological hydrogen production',
                'Biomass-waste gasification without CCUS',
            ],
        },
        'natural_hydrogen': {
            'category.1': ['Hydrogen'],
            'category.2': ['Production'],
            'category.3': ['Geologic hydrogen'],
        },
        'unabated_fossil': {'tech.final.name': []},
        'low_carbon_unspecified': {'tech.final.name': []},
        'infrastructure': {
            'category.1': ['Hydrogen'],
            'category.2': ['H₂ infrastructure'],
        },
        'power_generation': {
            'category.1': ['Hydrogen'],
            'category.2': ['Power generation'],
        },
        'hydrogen_based_fuels': {
            'tech.final.name': [
                'Biological CO2 methanation',
                'CO2 and h2o co-electrolysis',
                'CO₂-to-liquids (integrated RWGS + fischer–tropsch process)',
                'Chemical methanation',
                'Direct CO2 to dimethyl ether',
                'Haber-bosch process using electrolytic hydrogen',
                'Haber-bosch process using hydrogen produced with full CCUS',
                'Haber-bosch process using hydrogen produced with partial CCUS',
                'Low pressure haber-bosch process',
                'Methanol-to-jet',
                'Solar fuels',
            ],
        },
        'transport_use': {
            'category.1': ['Transport'],
            'Hydrogen.cc': [
                'Direct use',
                'Distribution',
                'Hydrogen-based fuels storage',
                'Hydrogen-based fuels use',
                'Storage',
            ],
        },
        'industrial_use': {
            'category.1': ['Industry'],
            'Hydrogen.cc': ['Direct use'],
        },
    },
    'CCS': {
        'direct_air_capture': {
            'category.1': ['Carbon Capture and Storage'],
            'category.2': ['CO2 capture'],
        },
        'power_capture': {
            'tech.final.name': [
                'Chemical looping combustion (coal)',
                'Oxy-fuelling (coal)',
                'Post-combustion: chemical absorption (coal with CCUS)',
                'Post-combustion: chemical absorption (natural gas with CCUS)',
                'Post-combustion: membranes polymeric (coal with CCUS)',
                'Post-combustion: solid adsorption (coal with CCUS)',
                'Pre-combustion: physical absorption (coal with CCUS)',
                'Supercritical CO2 cycle (coal with CCUS)',
                'Supercritical CO2 cycle (natural gas with CCUS)',
            ],
        },
        'industrial_capture': {
            'tech.final.name': [
                'Advanced acid gas removal',
                'Autothermal reforming - gas heated reformer',
                'Autothermal reforming - single reformer',
                'Biomass-waste gasification with CCUS',
                'Boilers fitted with CO2 capture',
                'CO2 capture in the bayer process',
                'Calcium looping carbon capture',
                'Chemical absorption (full capture rates)',
                'Chemical absorption, partial capture rates (less than 50%)',
                'Chemical looping (hydrogen production)',
                'Coal gasification - high capture rates',
                'Coal gasification - partial capture rates',
                'Cryogenic carbon capture',
                'DRI production based on chemical absorption carbon capture',
                'DRI production based on physical adsorption carbon capture',
                'Direct separation carbon capture',
                'Haber-bosch process using hydrogen produced with full CCUS',
                'Haber-bosch process using hydrogen produced with partial CCUS',
                'High value chemicals production using chemical absorption',
                'High value chemicals production using physical absorption',
                'Membrane separation carbon capture',
                'Methanol production with CO₂ capture (high rates)',
                'Methanol production with CO₂ capture (low rates)',
                'Novel physical adsorption carbon capture (silica or organic-based)',
                'Oxy-fuelling and carbon capture',
                'Oxy-fuelling carbon capture (cracking)',
                'Partial oxidation with CCUS',
                'Post-combustion carbon capture (cracking)',
                'Post-combustion carbon capture (process heater)',
                'Primary smelting with integrated CO2 capture',
                'Smelting with integrated CO2 capture',
                'Sorption enhanced steam reforming',
                'Steam reforming - high capture rates',
                'Steam reforming - partial capture',
            ],
        },
        'bioenergy_capture': {
            'category.1': ['Renewables'],
            'CCUS.cc': ['Capture'],
        },
        'transport': {
            'category.1': ['Carbon Capture and Storage'],
            'category.2': ['CO2 transport'],
        },
        'geological_storage': {
            'tech.final.name': [
                'Advanced CO2 storage monitoring technologies',
                'CO2 storage in depleted oil and gas reservoir',
                'CO2 storage in saline formation',
            ],
        },
        'mineral_storage': {
            'category.1': ['Carbon Capture and Storage'],
            'category.2': ['CO2 storage'],
            'category.3': ['Mineral storage'],
        },
        'enhanced_oil_recovery': {'tech.final.name': ['CO2-enhanced oil recovery']},
        'utilisation': {'CCUS.cc': ['Utilisation']},
    },
    'Biofuel': {
        'conventional_ethanol': {
            'tech.final.name': [
                'Sugar/starch fermentation',
                'Sugar/starch fermentation with CO2 capture',
            ],
        },
        'cellulosic_ethanol': {
            'tech.final.name': [
                'Lignocellulosic biomass fermentation',
                'Lignocellulosic biomass fermentation with CO2 capture',
            ],
        },
        'biodiesel': {'tech.final.name': ['Transesterification of oils and fats']},
        'hvo_hefa': {'tech.final.name': ['Hydrotreating of oil/fats to HVOs or HEFAs']},
        'alcohol_to_jet': {
            'tech.final.name': ['Biomethanol-to-jet', 'Ethanol / butanol-to-jet'],
        },
        'biogas_biomethane': {
            'tech.final.name': [
                'Anaerobic digestion and upgrading with CCUS',
                'Anaerobic digestion and upgrading with CO2 venting',
                'Anaerobic digestion with biological methanation',
                'Anaerobic digestion with chemical methanation',
                'Anaerobic digestion without biogas upgrading',
                'Biomass gasification - small-scale (bio-syngas)',
                'Biomass gasification and biological methanation',
                'Biomass gasification and catalytic methanation',
                'Biomass gasification and methanation with CCUS',
            ],
        },
        'gasification_to_liquids': {
            'category.1': ['Renewables'],
            'category.2': ['Bioenergy'],
            'category.3': ['Thermochemical conversion'],
            'category.4': ['Gasification to liquid fuels'],
        },
        'pyrolysis_and_liquefaction': {
            'tech.final.name': [
                'Biomass pyrolysis',
                'Hydrothermal liquefaction and upgrading of biomass',
            ],
        },
        'biomass_power_heat': {'Bioenergy.cc': ['Biomass use']},
        'sustainable_aviation_fuel': {'tech.final.name': []},
        'algae': {'tech.final.name': []},
    },
    'Geothermal': {
        'conventional_power': {
            'category.1': ['Renewables'],
            'category.2': ['Geothermal'],
            'category.3': ['Surface operations for geothermal energy'],
        },
        'next_generation': {
            'category.1': ['Renewables'],
            'category.2': ['Geothermal'],
            'category.3': ['Subsurface operations for geothermal energy'],
            'category.4': ['Next-generation geothermal energy systems'],
        },
        'drilling': {
            'category.1': ['Renewables'],
            'category.2': ['Geothermal'],
            'category.3': ['Subsurface operations for geothermal energy'],
            'category.4': ['Drilling'],
        },
        'ground_source_heat_pumps': {
            'tech.final.name': [
                'Inclined or deep horizontal wells heat pump',
                'Shallow ground-source heat pump',
            ],
        },
        'direct_heat': {'tech.final.name': []},
    },
    'Tidal': {
        'tidal_stream': {'tech.final.name': ['Tidal stream (ocean current energy)']},
        'tidal_range': {'tech.final.name': ['Tidal range']},
        'wave': {'tech.final.name': ['Wave energy converters (wave energy)']},
        'ocean_thermal': {'tech.final.name': ['Ocean thermal']},
        'salinity_gradient': {'tech.final.name': ['Salinity gradient (blue energy)']},
    },
    'Storage': {
        'lithium_ion': {'tech.final.name': ['Lithium-ion storage battery']},
        'sodium_ion': {'tech.final.name': ['Sodium-ion storage battery']},
        'flow_battery': {'tech.final.name': ['Redox flow storage battery']},
        'other_battery': {
            'tech.final.name': [
                'AI assisted battery material screening',
                'All solid-state battery',
                'Almost solid-state battery',
                'Batteries using hybrid chemistries',
                'Metal-air storage battery',
                'Semi solid-state battery',
                'Zinc-manganese oxide storage battery',
            ],
        },
        'pumped_storage': {'tech.final.name': ['Pumped storage']},
        'compressed_air': {'tech.final.name': ['Compressed air energy storage']},
        'other_mechanical': {
            'tech.final.name': [
                'Flywheel',
                'Gravity-based storage (excluding pumped hydro)',
                'Liquid CO₂ energy storage',
                'Liquid air energy storage',
            ],
        },
        'thermal_storage': {
            'category.1': ['Energy networks and storage'],
            'category.2': ['Thermal storage', 'Thermal storage for buildings and district heating'],
        },
        'battery_recycling': {
            'category.1': ['Critical minerals'],
            'category.2': ['E-waste recycling'],
            'category.3': ['Battery recycling'],
        },
    },
    'EVs': {
        'charging': {
            'category.1': ['Transport'],
            'category.2': ['Road transport'],
            'category.3': ['EV charging equipment'],
        },
        'fast_charging': {
            'tech.final.name': [
                'Fast and ultra-fast charging',
                'Fast charging',
                'Ultra-fast charging  (>1 MW)',
            ],
        },
        'battery_swapping': {
            'category.1': ['Transport'],
            'category.2': ['Road transport'],
            'category.3': ['EV charging equipment'],
            'category.4': ['Battery swapping'],
        },
        'smart_charging_v2g': {
            'category.1': ['Transport'],
            'category.2': ['Road transport'],
            'category.3': ['EV charging equipment'],
            'category.4': ['Demand response'],
        },
        'electric_road_systems': {
            'category.1': ['Transport'],
            'category.2': ['Road transport'],
            'category.3': ['EV charging equipment'],
            'category.4': ['Dynamic charging or electric road system'],
        },
        'passenger_vehicles': {'tech.final.name': ['Battery electric passenger car']},
        'commercial_vehicles': {
            'tech.final.name': [
                'Battery electric light commercial vehicle',
                'Battery electric truck',
                'Battery electric urban transit bus',
            ],
        },
        'ev_batteries': {
            'category.1': ['Transport'],
            'category.2': ['Road transport'],
            'category.3': ['Electric vehicle battery (electrical storage)'],
        },
        'battery_materials': {
            'category.1': ['Critical minerals'],
            'category.2': ['Mineral processing and refining', 'Mining and extraction'],
        },
        'ev_fluids': {'tech.final.name': []},
    },
    'Nature': {
        'reforestation': {'tech.final.name': []},
        'avoided_deforestation': {'tech.final.name': []},
        'blue_carbon': {'tech.final.name': []},
        'agriculture_and_soil': {'tech.final.name': []},
        'credit_trading': {'tech.final.name': []},
        'offset_products': {'tech.final.name': []},
    },
    'Nuclear': {
        'large_reactors': {
            'category.1': ['Nuclear'],
            'category.2': ['Fission'],
            'category.3': ['Large reactors'],
        },
        'small_modular_reactors': {
            'category.1': ['Nuclear'],
            'category.2': ['Fission'],
            'category.3': ['Small modular reactors'],
        },
        'fuel_cycle': {
            'tech.final.name': [
                'Fuel fabrication: TRISO fuel for high temperature reactors',
                'Fuel reprocessing (PUREX process and new advanced techniques)',
                'HALEU fuel cycle',
                'In-situ leaching (ISL)',
                'Radioactive waste disposal',
                'Thorium fuel cycle',
                'Uranium conversion',
                'Uranium enrichment: conventional techniques',
                'Uranium enrichment: laser-based techniques',
            ],
        },
    },
    'Fusion': {
        'magnetic_confinement': {
            'category.1': ['Nuclear'],
            'category.2': ['Fusion'],
            'category.3': ['Magnetic confinement fusion'],
        },
        'inertial_confinement': {
            'category.1': ['Nuclear'],
            'category.2': ['Fusion'],
            'category.3': ['Inertial confinement fusion'],
        },
        'alternative_approaches': {
            'category.1': ['Nuclear'],
            'category.2': ['Fusion'],
            'category.3': ['Alternative approaches'],
        },
    },
    'Coal': {
        'thermal_coal': {'tech.final.name': []},
        'metallurgical_coal': {'tech.final.name': []},
        'coal_power': {'tech.final.name': []},
        'coal_conversion': {'tech.final.name': []},
        'methane_abatement': {
            'category.1': ['Fossil fuels'],
            'category.2': ['Coal'],
            'category.3': ['Methane emissions monitoring and abatement'],
        },
    },
    'Oil': {
        'onshore': {'tech.final.name': []},
        'shallow_water': {'tech.final.name': []},
        'deepwater': {'tech.final.name': []},
        'arctic': {'tech.final.name': []},
        'enhanced_recovery': {'tech.final.name': []},
        'methane_abatement': {
            'category.1': ['Fossil fuels'],
            'category.2': ['Oil and gas'],
            'category.3': ['Methane emissions monitoring and abatement'],
        },
    },
    'Natural gas': {
        'upstream_production': {'tech.final.name': []},
        'lng': {'tech.final.name': []},
        'pipeline': {'tech.final.name': []},
        'gas_power': {'tech.final.name': []},
        'transport_fuel': {'tech.final.name': []},
        'ngls': {'tech.final.name': []},
        'methane_abatement': {
            'category.1': ['Fossil fuels'],
            'category.2': ['Oil and gas'],
            'category.3': ['Methane emissions monitoring and abatement'],
        },
    },
    'Refining': {
        'fuels_refining': {'tech.final.name': []},
        'lubricants': {'tech.final.name': []},
        'biorefinery_conversion': {'tech.final.name': []},
        'co_processing': {'tech.final.name': []},
        'integrated_chemicals': {'tech.final.name': []},
    },
    'Petrochemicals': {
        'olefins': {'tech.final.name': []},
        'aromatics': {'tech.final.name': []},
        'polymers': {'tech.final.name': []},
        'specialty_chemicals': {'tech.final.name': []},
        'chemical_recycling': {
            'category.1': ['Industry'],
            'category.2': ['Chemicals'],
            'category.3': ['Recycling'],
        },
    },
    'Unconventional': {
        'tight_oil': {'tech.final.name': []},
        'tight_gas': {'tech.final.name': []},
        'oil_sands': {'tech.final.name': []},
        'heavy_oil': {'tech.final.name': []},
        'coalbed_methane': {'tech.final.name': []},
    },
}

TERMS_ALL = TermCollection(
    [
          Term(alias = 'Solar', regex = SOLAR_REGEX, 
               colour = 'gold', 
               prompt_alias='solar power',
               group='decarbonisation'),
          Term(alias = 'Wind', regex = WIND_REGEX, 
               colour = 'powderblue',
               prompt_alias='wind power',
               group='decarbonisation'),
          Term(alias = 'Hydrogen (green)', regex = GREEN_HYDROGEN_REGEX, 
               colour = 'turquoise', dash=True, edgecolour='chartreuse', hatch='....',
               prompt_alias='green hydrogen',
               group='decarbonisation'),
          Term(alias = 'Hydrogen (blue)', regex = BLUE_HYDROGEN_REGEX, 
               colour = 'turquoise', dash=True, edgecolour='blue', hatch='....',
               prompt_alias='blue hydrogen',
               group='decarbonisation'),
          Term(alias = 'Hydrogen', regex = HYDROGEN_REGEX, 
               colour = 'turquoise',
               prompt_alias='hydrogen',
               group='decarbonisation'),
          Term(alias = 'CCS', regex = CCS_REGEX, 
               colour = 'dimgray',
               prompt_alias='carbon capture and storage',
               group='decarbonisation'),
          Term(alias = 'Biofuel', regex = BIOFUEL_REGEX, 
               colour = 'tab:brown', dash=True, edgecolour='peru', hatch='....',
               prompt_alias='biofuel',
               group='decarbonisation'),
          Term(alias = 'Biogas', regex = BIOGAS_REGEX, 
               colour = 'tab:brown', dash=True, edgecolour='lightgrey', hatch='....',
               prompt_alias='biogas',
               group='decarbonisation'),
          Term(alias = 'Geothermal', regex = GEOTHERMAL_REGEX, 
               colour = 'tab:red',
               prompt_alias='geothermal energy',
               group='decarbonisation'),
          Term(alias = 'Tidal', regex = TIDAL_REGEX, 
               colour = 'steelblue',
               prompt_alias='tidal energy',
               group='decarbonisation'),

          Term(alias = 'Storage', regex = STORAGE_REGEX, 
               colour = 'pink',
               prompt_alias='energy storage',
               group='decarbonisation'),
          Term(alias = 'EVs', regex = EVS_REGEX, 
               colour = 'lightgrey',
               prompt_alias='electric vehicles',
               group='decarbonisation'),
          Term(alias = 'Nature', regex = NATURE_REGEX,
               colour = 'darkgreen',
               prompt_alias='nature-based solutions',
               group='decarbonisation'),

          Term(alias = 'Nuclear (conventional)', regex = NUCLEAR_CONVENTIONAL_REGEX,
               colour = 'tab:olive',
               prompt_alias='conventional nuclear power',
               group='decarbonisation'),
          Term(alias = 'Nuclear (SMR)', regex = NUCLEAR_SMR_REGEX, 
               colour = 'tab:olive', dash=True, edgecolour='tab:grey', hatch='....',
               prompt_alias='small modular reactors',
               group='decarbonisation'),
          Term(alias = 'Nuclear fusion', regex = NUCLEAR_FUSION_REGEX, 
               colour = 'tab:orange',
               prompt_alias='nuclear fusion',
               group='decarbonisation'),
     ],
     folder_name='all-terms'
)

TERMS_SUMMARY = TermCollection(
    [
        Term(alias = 'Solar', regex = SOLAR_REGEX, 
             colour = 'gold',
             prompt_alias='solar power',
             group='decarbonisation'),
        Term(alias = 'Wind', regex = WIND_REGEX, 
             colour = 'powderblue',
             prompt_alias='wind power',
             group='decarbonisation'),
        Term(alias = 'Hydrogen', regex = f'{GREEN_HYDROGEN_REGEX}|{BLUE_HYDROGEN_REGEX}|{HYDROGEN_REGEX}', 
             colour = 'turquoise',
             prompt_alias='hydrogen',
             group='decarbonisation'),
        Term(alias = 'CCS', regex = CCS_REGEX, 
             colour = 'dimgray',
             prompt_alias='carbon capture and storage',
             group='decarbonisation'),
        Term(alias = 'Bioenergy', regex = f'{BIOFUEL_REGEX}|{BIOGAS_REGEX}', 
             colour = 'tab:brown',
             prompt_alias='bioenergy',
             group='decarbonisation'),
        Term(alias = 'Geothermal', regex = GEOTHERMAL_REGEX, 
             colour = 'tab:red',
             prompt_alias='geothermal energy',
             group='decarbonisation'),
        Term(alias = 'Tidal', regex = TIDAL_REGEX, 
             colour = 'steelblue',
             prompt_alias='tidal energy',
             group='decarbonisation'),

        Term(alias = 'Storage', regex = STORAGE_REGEX, 
             colour = 'pink',
             prompt_alias='energy storage',
             group='decarbonisation'),
        Term(alias = 'EVs', regex = EVS_REGEX, 
             colour = 'lightgrey',
             prompt_alias='electric vehicles',
             group='decarbonisation'),
        Term(alias = 'Nature', regex = NATURE_REGEX, 
             colour = 'darkgreen',
             prompt_alias='nature-based solutions',
             group='decarbonisation'),

        Term(alias = 'Nuclear', regex = f'{NUCLEAR_CONVENTIONAL_REGEX}|{NUCLEAR_SMR_REGEX}', 
             colour = 'tab:olive',
             prompt_alias='nuclear energy',
             group='decarbonisation'),
        Term(alias = 'Fusion', regex = NUCLEAR_FUSION_REGEX, 
             colour = 'tab:orange',
             prompt_alias='nuclear fusion',
             group='decarbonisation')
    ], 
    folder_name='summary-terms'
)

TERMS_SIMPLE = TermCollection(
    [
        Term(alias = 'Solar', regex = r'solar|\bPV\b|\bP\.V\.\b|photovoltaic|\bCSP\b|\bC\.S\.P\.\b', 
             colour = 'gold',
             prompt_alias='solar power',
             group='decarbonisation',
             variants = TECHNOLOGY_VARIANTS_IEA['Solar'],
             variants_iea_indexer = TECHNOLOGY_VARIANTS_IEA_INDS['Solar']),
        Term(alias = 'Wind', regex = r'\bwinds?\b(?!\s+(down|up))',
             colour = 'powderblue',
             prompt_alias='wind power',
             group='decarbonisation',
             variants=TECHNOLOGY_VARIANTS_IEA['Wind'],
             variants_iea_indexer=TECHNOLOGY_VARIANTS_IEA_INDS['Wind']),
        Term(alias = 'Hydrogen', regex = r'hydrogen|h2\b', 
             colour = 'turquoise',
             prompt_alias='hydrogen',
             group='decarbonisation',
             variants = TECHNOLOGY_VARIANTS_IEA['Hydrogen'],
             variants_iea_indexer = TECHNOLOGY_VARIANTS_IEA_INDS['Hydrogen']),
        Term(alias = 'CCS', regex = r'carbon\s+capture|\bccs\b|\bccus\b|\bbeccs\b|carbon\s+sequestration|co[2₂]\s+(capture|storage|sequestration|injection)|direct\s+air\s+capture|\bDAC\b',
             colour = 'dimgray',
             prompt_alias = 'carbon capture and storage',
             group = 'decarbonisation',
             variants =TECHNOLOGY_VARIANTS_IEA['CCS'],
             variants_iea_indexer = TECHNOLOGY_VARIANTS_IEA_INDS['CCS']),
        Term(alias = 'Biofuel', regex = r'bio-?(fuel|diesel|ethanol|methane|gas|mass)|renewable\s+(natural\s+gas|diesel)|\bRNG\b|\bR\.N\.G\.\b|\bSAF\b|sustainable\s+aviation\s+fuel|hydrotreated\s+vegetable\s+oil|\bHVO\b',
             colour = 'tab:brown',
             prompt_alias='biofuel',
             group='decarbonisation',
             variants=TECHNOLOGY_VARIANTS_IEA['Biofuel'],
             variants_iea_indexer=TECHNOLOGY_VARIANTS_IEA_INDS['Biofuel']),
        Term(alias = 'Geothermal', regex = r'geothermal|ground.source|ground\s+source', 
             colour = 'tab:red',
             prompt_alias='geothermal energy',
             group='decarbonisation',
             variants=TECHNOLOGY_VARIANTS_IEA['Geothermal'],
             variants_iea_indexer=TECHNOLOGY_VARIANTS_IEA_INDS['Geothermal']),
        Term(alias = 'Tidal', regex = r'tidal|wave\s+(energy|power|farm|converter|device|project)|ocean\s+energy|\bOTEC\b',
             colour = 'steelblue',
             prompt_alias='tidal energy',
             group='decarbonisation',
             variants=TECHNOLOGY_VARIANTS_IEA['Tidal'],
             variants_iea_indexer=TECHNOLOGY_VARIANTS_IEA_INDS['Tidal']),
        Term(alias = 'Storage', regex = r'batter(y|ies)\s+(storage|system|technolog|pack|cell|manufactur|plant|project|capacity|chemistr)|(advanced|lithium(.ion)?|solid.state|flow|redox|vanadium|sodium.ion|nickel.metal|nimh|lead.acid)\s+batter(y|ies)|energy\s+storage|\b(BESS|B\.E\.S\.S\.|CAES|C\.A\.E\.S\.|LDES)\b|compressed\s+air|pumped.hydro|grid.scale\s+storage',
             colour = 'pink',
             prompt_alias='energy storage',
             group='decarbonisation',
             variants=TECHNOLOGY_VARIANTS_IEA['Storage'],
             variants_iea_indexer=TECHNOLOGY_VARIANTS_IEA_INDS['Storage']),
        Term(alias = 'EVs', regex = r'\b(electric\s+vehicle|EV|E\.V\.|BEV|PHEV)s?\b|plug.?in\s+hybrid|charg(ing|e)\s+(point|station|network|infrastructure|hub)|vehicle.to.grid|\bV2G\b',
             colour = 'lightgrey',
             prompt_alias='electric vehicles',
             group='decarbonisation',
             variants=TECHNOLOGY_VARIANTS_IEA['EVs'],
             variants_iea_indexer=TECHNOLOGY_VARIANTS_IEA_INDS['EVs']),
        Term(alias = 'Nature', regex = r'\b(nature.based|nature\s+based|(re|af)forestation)\b|carbon\s+(credit|offset)|natural\s+climate\s+solution|voluntary\s+carbon\s+market|\bVCM\b|REDD\+?|blue\s+carbon|forest\s+carbon',
             colour = 'darkgreen',
             prompt_alias='nature-based solutions',
             group='decarbonisation',
             variants=TECHNOLOGY_VARIANTS_IEA['Nature'],
             variants_iea_indexer=TECHNOLOGY_VARIANTS_IEA_INDS['Nature']),
        Term(alias = 'Nuclear', regex = r'nuclear\s+(power|plants?|energy|reactor|fleet|fission|programme|program|capacity|technology)|(small|advanced)\s+modular\s+reactor',
             colour = 'tab:olive',
             prompt_alias='nuclear energy',
             group='decarbonisation',
             variants=TECHNOLOGY_VARIANTS_IEA['Nuclear'],
             variants_iea_indexer=TECHNOLOGY_VARIANTS_IEA_INDS['Nuclear']),
        Term(alias = 'Fusion', regex = r'nuclear\s+fusion|fusion\s+(energy|power|reactor|plant|research)|(magnetic|inertial|confinement)\s+fusion|tokamak|stellarator',
             colour = 'tab:orange',
             prompt_alias='nuclear fusion',
             group='decarbonisation',
             variants=TECHNOLOGY_VARIANTS_IEA['Fusion'],
             variants_iea_indexer=TECHNOLOGY_VARIANTS_IEA_INDS['Fusion']),

        # Fossil fuel technologies, for comparison against the decarbonisation terms above.
        Term(alias = 'Coal', regex = r'\bcoal\b|lignite|anthracite|coal.fired|thermal\s+coal|metallurgical\s+coal|coking\s+coal|coal.to.(liquid|gas)|\bCTL\b',
             colour = 'black',
             prompt_alias='coal',
             group='fossil',
             variants=TECHNOLOGY_VARIANTS_IEA['Coal'],
             variants_iea_indexer=TECHNOLOGY_VARIANTS_IEA_INDS['Coal']),
        Term(alias = 'Oil', regex = r'crude\s+oil|\bcrude\b|petroleum\s+(product|refin|liquid)|oil\s+(production|field|reserves|price|demand|supply|output|well|discover)|\bbbls?\b|barrels?\s+of\s+oil(?!\s+equivalent)',
             colour = 'saddlebrown',
             prompt_alias='crude oil',
             group='fossil',
             variants=TECHNOLOGY_VARIANTS_IEA['Oil'],
             variants_iea_indexer=TECHNOLOGY_VARIANTS_IEA_INDS['Oil']),
        Term(alias = 'Natural gas', regex = r'natural\s+gas|\bLNG\b|\bCNG\b|\bNGLs?\b|gas.fired|\bCCGT\b|shale\s+gas|associated\s+gas|gas\s+(production|field|reserves|pipeline|processing|liquefaction|market|demand|supply|price|well)',
             colour = 'indianred',
             prompt_alias='natural gas',
             group='fossil',
             variants=TECHNOLOGY_VARIANTS_IEA['Natural gas'],
             variants_iea_indexer=TECHNOLOGY_VARIANTS_IEA_INDS['Natural gas']),
        Term(alias = 'Refining', regex = r'refin(ery|eries|ing)|\bcrackers?\b|hydrocracker|catalytic\s+crack|distillation\s+unit|throughput\s+capacity',
             colour = 'darkslateblue',
             prompt_alias='oil refining',
             group='fossil',
             variants=TECHNOLOGY_VARIANTS_IEA['Refining'],
             variants_iea_indexer=TECHNOLOGY_VARIANTS_IEA_INDS['Refining']),
        Term(alias = 'Petrochemicals', regex = r'petrochemical|\bolefins?\b|\baromatics?\b|polyethylene|polypropylene|\bPTA\b|ethylene|propylene|\bmonomer',
             colour = 'rebeccapurple',
             prompt_alias='petrochemicals',
             group='fossil',
             variants=TECHNOLOGY_VARIANTS_IEA['Petrochemicals'],
             variants_iea_indexer=TECHNOLOGY_VARIANTS_IEA_INDS['Petrochemicals']),
        Term(alias = 'Unconventional', regex = r'oil\s+sands|\bshale\b|tight\s+oil|tight\s+gas|fracking|hydraulic\s+fractur|\bbitumen\b|heavy\s+oil|coalbed\s+methane|\bCBM\b|synthetic\s+crude',
             colour = 'darkkhaki',
             prompt_alias='unconventional oil and gas',
             group='fossil',
             variants=TECHNOLOGY_VARIANTS_IEA['Unconventional'],
             variants_iea_indexer=TECHNOLOGY_VARIANTS_IEA_INDS['Unconventional'])
    ],
    folder_name = 'simple-terms'
)