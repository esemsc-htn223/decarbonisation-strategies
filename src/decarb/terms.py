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
             group='decarbonisation'),
        Term(alias = 'Wind', regex = r'\bwinds?\b(?!\s+(down|up))',
             colour = 'powderblue',
             prompt_alias='wind power',
             group='decarbonisation'),
        Term(alias = 'Hydrogen', regex = r'hydrogen|h2', 
             colour = 'turquoise',
             prompt_alias='hydrogen',
             group='decarbonisation'),
        Term(alias = 'CCS', regex = r'carbon\s+capture|\bccs\b|\bccus\b|\bbeccs\b|carbon\s+sequestration|co2\s+(capture|storage|sequestration|injection)|direct\s+air\s+capture|\bDAC\b',
             colour = 'dimgray',
             prompt_alias='carbon capture and storage',
             group='decarbonisation'),
        Term(alias = 'Biofuel', regex = r'bio-?(fuel|diesel|ethanol|methane|gas|mass)|renewable\s+(natural\s+gas|diesel)|\bRNG\b|\bR\.N\.G\.\b|\bSAF\b|sustainable\s+aviation\s+fuel|hydrotreated\s+vegetable\s+oil|\bHVO\b',
             colour = 'tab:brown',
             prompt_alias='biofuel',
             group='decarbonisation'),
        Term(alias = 'Geothermal', regex = r'geothermal|ground\.source|ground\s+source', 
             colour = 'tab:red',
             prompt_alias='geothermal energy',
             group='decarbonisation'),
        Term(alias = 'Tidal', regex = r'tidal|wave\s+(energy|power|farm|converter|device|project)|ocean\s+energy|\bOTEC\b',
             colour = 'steelblue',
             prompt_alias='tidal energy',
             group='decarbonisation'),
        Term(alias = 'Storage', regex = r'batter(y|ies)\s+(storage|system|technolog|pack|cell|manufactur|plant|project|capacity|chemistr)|(advanced|lithium(.ion)?|solid.state|flow|redox|vanadium|sodium.ion|nickel.metal|nimh|lead.acid)\s+batter(y|ies)|energy\s+storage|\b(BESS|B\.E\.S\.S\.|CAES|C\.A\.E\.S\.|LDES)\b|compressed\s+air|pumped.hydro|grid.scale\s+storage',
             colour = 'pink',
             prompt_alias='energy storage',
             group='decarbonisation'),
        Term(alias = 'EVs', regex = r'\b(electric\s+vehicle|EV|E\.V\.|BEV|PHEV)s?\b|plug.?in\s+hybrid|charg(ing|e)\s+(point|station|network|infrastructure|hub)|vehicle.to.grid|\bV2G\b',
             colour = 'lightgrey',
             prompt_alias='electric vehicles',
             group='decarbonisation'),
        Term(alias = 'Nature', regex = r'\b(nature.based|nature\s+based|(re|af)forestation)\b|carbon\s+(credit|offset)|natural\s+climate\s+solution|voluntary\s+carbon\s+market|\bVCM\b|REDD\+?|blue\s+carbon|forest\s+carbon',
             colour = 'darkgreen',
             prompt_alias='nature-based solutions',
             group='decarbonisation'),
        Term(alias = 'Nuclear', regex = r'nuclear\s+(power|plants?|energy|reactor|fleet|fission|programme|program|capacity|technology)|(small|advanced)\s+modular\s+reactor',
             colour = 'tab:olive',
             prompt_alias='nuclear energy',
             group='decarbonisation'),
        Term(alias = 'Fusion', regex = r'nuclear\s+fusion|fusion\s+(energy|power|reactor|plant|research)',
             colour = 'tab:orange',
             prompt_alias='nuclear fusion',
             group='decarbonisation'),

        # Fossil fuel technologies, for comparison against the decarbonisation terms above.
        Term(alias = 'Coal', regex = r'\bcoal\b|lignite|anthracite|coal.fired|thermal\s+coal|metallurgical\s+coal|coking\s+coal|coal.to.(liquid|gas)|\bCTL\b',
             colour = 'black',
             prompt_alias='coal',
             group='fossil'),
        Term(alias = 'Oil', regex = r'crude\s+oil|\bcrude\b|petroleum\s+(product|refin|liquid)|oil\s+(production|field|reserves|price|demand|supply|output|well|discover)|\bbbls?\b|barrels?\s+of\s+oil(?!\s+equivalent)',
             colour = 'saddlebrown',
             prompt_alias='crude oil',
             group='fossil'),
        Term(alias = 'Natural gas', regex = r'natural\s+gas|\bLNG\b|\bCNG\b|\bNGLs?\b|gas.fired|\bCCGT\b|shale\s+gas|associated\s+gas|gas\s+(production|field|reserves|pipeline|processing|liquefaction|market|demand|supply|price|well)',
             colour = 'indianred',
             prompt_alias='natural gas',
             group='fossil'),
        Term(alias = 'Refining', regex = r'refin(ery|eries|ing)|\bcrackers?\b|hydrocracker|catalytic\s+crack|distillation\s+unit|throughput\s+capacity',
             colour = 'darkslateblue',
             prompt_alias='oil refining',
             group='fossil'),
        Term(alias = 'Petrochemicals', regex = r'petrochemical|\bolefins?\b|\baromatics?\b|polyethylene|polypropylene|\bPTA\b|ethylene|propylene|\bmonomer',
             colour = 'rebeccapurple',
             prompt_alias='petrochemicals',
             group='fossil'),
        Term(alias = 'Unconventional', regex = r'oil\s+sands|\bshale\b|tight\s+oil|tight\s+gas|fracking|hydraulic\s+fractur|\bbitumen\b|heavy\s+oil|coalbed\s+methane|\bCBM\b|synthetic\s+crude',
             colour = 'darkkhaki',
             prompt_alias='unconventional oil and gas',
             group='fossil')
    ],
    folder_name = 'simple-terms'
)