__all__ = ['Term', 'TermCollection', 'TERMS_ALL', 'TERMS_SUMMARY', 'TERMS_SIMPLE']

import os
import matplotlib.pyplot as plt
from typing import Iterable
from dataclasses import dataclass
import uuid

from ..util import TERMS_DIR, logger

@dataclass
class Term:
     alias: str
     regex: str
     colour: str
     prompt_alias: str|None = None

     #linestyle: str|None = None
     dash: bool = False
     marker: str|None = None
     hatch: str|None = None
     edgecolour: str|None = None
     dash_ls = (0, (5, 10)) # custom dashed linestyle for terms with dash=True

     def __post_init__(self):
          if self.prompt_alias is None:
               self.prompt_alias = self.alias
          self.prompt_fname = self.alias.lower().replace(' ', '-').replace('(', '').replace(')', '') + '.txt'
     
    
     def __str__(self):
          return self.alias
     
     def set_collection(self, collection: 'TermCollection'):
          if hasattr(self, 'collection'):
               raise ValueError(f'Term {self.alias} is already part of a collection. Cannot set collection again.')
          self.collection = collection
    
    

class TermCollection:
     def __init__(self, terms: Iterable[Term], folder_name: str):

          # check that all aliases are unique
          aliases = [term.alias for term in terms]
          if len(aliases) != len(set(aliases)):
               raise ValueError('All term aliases must be unique')

          for term in terms:
               term.set_collection(self)

          self.terms = list(terms)
          self.folder_name = folder_name.replace(' ', '-').lower()

          self.folder = os.path.join(TERMS_DIR, self.folder_name)

          if not os.path.exists(self.folder):
               os.makedirs(self.folder)
          if not os.path.exists(os.path.join(self.folder, 'terms.txt')):
               with open(os.path.join(self.folder, 'terms.txt'), 'w') as f:
                    for term in self.terms:
                         f.write(f'{term.alias}: {term.regex}\n')

          self.uuid = 'terms-' + uuid.uuid5(uuid.NAMESPACE_OID, ''.join([term.alias + term.regex for term in self.terms])).hex

          logger.debug(f'CREATED TermCollection: folder_name={self.folder_name}, uuid={self.uuid}, terms={[term.alias for term in self.terms]}')

     def __iter__(self):
          return iter(self.terms)
    
     def __getitem__(self, key: str|int):
          if isinstance(key, int):
               return self.terms[key]
          for term in self.terms:
               if term.alias == key:
                   return term
               if term.alias.lower() == key.lower():
                  return term
          raise KeyError(f'Term with alias {key} not found')
    
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
               prompt_alias='solar power'),
          Term(alias = 'Wind', regex = WIND_REGEX, 
               colour = 'powderblue',
               prompt_alias='wind power'),
          Term(alias = 'Hydrogen (green)', regex = GREEN_HYDROGEN_REGEX, 
               colour = 'turquoise', dash=True, edgecolour='chartreuse', hatch='....',
               prompt_alias='green hydrogen'),
          Term(alias = 'Hydrogen (blue)', regex = BLUE_HYDROGEN_REGEX, 
               colour = 'turquoise', dash=True, edgecolour='blue', hatch='....',
               prompt_alias='blue hydrogen'),
          Term(alias = 'Hydrogen', regex = HYDROGEN_REGEX, 
               colour = 'turquoise',
               prompt_alias='hydrogen'),
          Term(alias = 'CCS', regex = CCS_REGEX, 
               colour = 'dimgray',
               prompt_alias='carbon capture and storage'),
          Term(alias = 'Biofuel', regex = BIOFUEL_REGEX, 
               colour = 'tab:brown', dash=True, edgecolour='peru', hatch='....',
               prompt_alias='biofuel'),
          Term(alias = 'Biogas', regex = BIOGAS_REGEX, 
               colour = 'tab:brown', dash=True, edgecolour='lightgrey', hatch='....',
               prompt_alias='biogas'),
          Term(alias = 'Geothermal', regex = GEOTHERMAL_REGEX, 
               colour = 'tab:red',
               prompt_alias='geothermal energy'),
          Term(alias = 'Tidal', regex = TIDAL_REGEX, 
               colour = 'steelblue',
               prompt_alias='tidal energy'),

          Term(alias = 'Storage', regex = STORAGE_REGEX, 
               colour = 'pink',
               prompt_alias='energy storage'),
          Term(alias = 'EVs', regex = EVS_REGEX, 
               colour = 'lightgrey',
               prompt_alias='electric vehicles'),
          Term(alias = 'Nature', regex = NATURE_REGEX, 
               colour = 'darkgreen',
               prompt_alias='nature-based solutions'),
          Term(alias = 'Efficiency', regex = EFFICIENCY_REGEX, 
               colour = 'fuchsia',
               prompt_alias='energy efficiency'),

          Term(alias = 'Nuclear (conventional)', regex = NUCLEAR_CONVENTIONAL_REGEX, 
               colour = 'tab:olive',
               prompt_alias='conventional nuclear power'),
          Term(alias = 'Nuclear (SMR)', regex = NUCLEAR_SMR_REGEX, 
               colour = 'tab:olive', dash=True, edgecolour='tab:grey', hatch='....',
               prompt_alias='small modular reactors'),
          Term(alias = 'Nuclear fusion', regex = NUCLEAR_FUSION_REGEX, 
               colour = 'tab:orange',
               prompt_alias='nuclear fusion'),
     ],
     folder_name='all-terms'
)

TERMS_SUMMARY = TermCollection(
    [
        Term(alias = 'Solar', regex = SOLAR_REGEX, 
             colour = 'gold',
             prompt_alias='solar power'),
        Term(alias = 'Wind', regex = WIND_REGEX, 
             colour = 'powderblue',
             prompt_alias='wind power'),
        Term(alias = 'Hydrogen', regex = f'{GREEN_HYDROGEN_REGEX}|{BLUE_HYDROGEN_REGEX}|{HYDROGEN_REGEX}', 
             colour = 'turquoise',
             prompt_alias='hydrogen'),
        Term(alias = 'CCS', regex = CCS_REGEX, 
             colour = 'dimgray',
             prompt_alias='carbon capture and storage'),
        Term(alias = 'Bioenergy', regex = f'{BIOFUEL_REGEX}|{BIOGAS_REGEX}', 
             colour = 'tab:brown',
             prompt_alias='bioenergy'),
        Term(alias = 'Geothermal', regex = GEOTHERMAL_REGEX, 
             colour = 'tab:red',
             prompt_alias='geothermal energy'),
        Term(alias = 'Tidal', regex = TIDAL_REGEX, 
             colour = 'steelblue',
             prompt_alias='tidal energy'),

        Term(alias = 'Storage', regex = STORAGE_REGEX, 
             colour = 'pink',
             prompt_alias='energy storage'),
        Term(alias = 'EVs', regex = EVS_REGEX, 
             colour = 'lightgrey',
             prompt_alias='electric vehicles'),
        Term(alias = 'Nature', regex = NATURE_REGEX, 
             colour = 'darkgreen',
             prompt_alias='nature-based solutions'),
        Term(alias = 'Efficiency', regex = EFFICIENCY_REGEX, 
             colour = 'fuchsia',
             prompt_alias='energy efficiency'),

        Term(alias = 'Nuclear', regex = f'{NUCLEAR_CONVENTIONAL_REGEX}|{NUCLEAR_SMR_REGEX}', 
             colour = 'tab:olive',
             prompt_alias='nuclear energy'),
        Term(alias = 'Fusion', regex = NUCLEAR_FUSION_REGEX, 
             colour = 'tab:orange',
             prompt_alias='nuclear fusion')
    ], 
    folder_name='summary-terms'
)

TERMS_SIMPLE = TermCollection(
    [
        Term(alias = 'Solar', regex = r'solar|\bPV\b|\bP\.V\.\b|photovoltaic|\bCSP\b|\bC\.S\.P\.\b', 
             colour = 'gold',
             prompt_alias='solar power'),
        Term(alias = 'Wind', regex = r'\bwind\b', 
             colour = 'powderblue',
             prompt_alias='wind power'),
        Term(alias = 'Hydrogen', regex = r'hydrogen|h2', 
             colour = 'turquoise',
             prompt_alias='hydrogen'),
        Term(alias = 'CCS', regex = r'(carbon\s+capture(\s+and\s+(storage|sequestration|utili[sz]ation))|ccs)', 
             colour = 'dimgray',
             prompt_alias='carbon capture and storage'),
        Term(alias = 'Biofuel', regex = r'bio-?(fuel|diesel|ethanol|methane|gas)|renewable\s+natural\s+gas|\bRNG\b|\bR\.N\.G\.\b', 
             colour = 'tab:brown',
             prompt_alias='biofuel'),
        Term(alias = 'Geothermal', regex = r'geothermal|ground\.source|ground\s+source', 
             colour = 'tab:red',
             prompt_alias='geothermal energy'),
        Term(alias = 'Tidal', regex = r'tidal|wave', 
             colour = 'steelblue',
             prompt_alias='tidal energy'),
        Term(alias = 'Storage', regex = r'\b(batter(y|ies)|BESS|B\.E\.S\.S\.|compressed\s+air|CAES|C\.A\.E\.S\.)\b', 
             colour = 'pink',
             prompt_alias='energy storage'),
        Term(alias = 'EVs', regex = r'\b(electric\s+vehicle|EV|E\.V\.)s?\b', 
             colour = 'lightgrey',
             prompt_alias='electric vehicles'),
        Term(alias = 'Nature', regex = r'\b(nature.based|nature\s+based|(re|a)forestation)\b', 
             colour = 'darkgreen',
             prompt_alias='nature-based solutions'),
        Term(alias = 'Nuclear', regex = r'nuclear\s+(power|plants?|energy|reactor|fleet|fission)|(small|advanced)\s+modular\s+reactor', 
             colour = 'tab:olive',
             prompt_alias='nuclear energy'),
        Term(alias = 'Fusion', regex = r'nuclear\s+fusion|fusion\s+(energy|power|reactor|plant|research)', 
             colour = 'tab:orange',
             prompt_alias='nuclear fusion')
    ],
    folder_name = 'simple-terms'
)