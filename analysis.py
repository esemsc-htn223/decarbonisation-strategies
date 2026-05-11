import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 0. Imports & Setup
    """)
    return


@app.cell
def _():
    import pandas as pd
    import re
    import os
    import pymupdf
    import marimo as mo
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from typing import Iterable
    from dataclasses import dataclass
    import uuid
    import logging

    pymupdf.TOOLS.mupdf_display_errors(on = False)
    pymupdf.TOOLS.mupdf_display_warnings(on = False)
    return (
        Iterable,
        dataclass,
        logging,
        mo,
        mpl,
        np,
        os,
        pd,
        plt,
        pymupdf,
        re,
        uuid,
    )


@app.cell
def _(COS, Iterable, dataclass, logging, os, pd, plt, search_files, uuid):
    logger = logging.getLogger('analysis')
    logger.setLevel(logging.DEBUG)

    logger_fh = logging.FileHandler(os.path.join('out', 'analysis.log'))
    logger_fh.setLevel(logging.DEBUG)
    logger_fh.set_name('analysis_debug')
    logger_fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    @dataclass
    class Term:
        alias: str
        regex: str
        colour: str

        #linestyle: str|None = None
        dash: bool = False
        marker: str|None = None
        hatch: str|None = None
        edgecolour: str|None = None
        dash_ls = (0, (5, 10)) # custom dashed linestyle for terms with dash=True

    class TermCollection:
        def __init__(self, terms: Iterable[Term], folder_name: str):

            # check that all aliases are unique
            aliases = [term.alias for term in terms]
            if len(aliases) != len(set(aliases)):
                raise ValueError('All term aliases must be unique')

            self.terms = list(terms)
            self.folder_name = folder_name.replace(' ', '-').lower()
            self.folder = os.path.join('out', self.folder_name)
        
            if not os.path.exists(self.folder):
                os.makedirs(self.folder)
                with open(os.path.join(self.folder, 'terms.txt'), 'w') as f:
                    for term in self.terms:
                        f.write(f'{term.alias}: {term.regex}\n')

            self.uuid = uuid.uuid5(uuid.NIL, ''.join([term.alias + term.regex for term in self.terms])).hex

            self.log('CREATED')

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
    
        def run_pipeline(self, 
                        count_type: str = 'all', norm_type: str = 'words',
                        reprocess: bool = False, save_csv: bool = False):

            report_types = {'annual': 'an', 'sustainability-report': 'sr'}

            dfs = {}
            dfs_unnormed = {}

            if not os.path.exists('out/mentions'):
                os.makedirs('out/mentions')
            if not os.path.exists('out/mentions-unnormed'):
                os.makedirs('out/mentions-unnormed')

            normed_f = lambda co, rt: f'out/mentions/{co}-{report_types[rt]}.csv'
            unnormed_f = lambda co, rt: f'out/mentions-unnormed/{co}-{report_types[rt]}.csv'

            for co in COS:
                for report_type in report_types:
                    folder = os.path.join('reports', co, report_type)

                    if reprocess or not os.path.exists(normed_f(co, report_type)):
                        dfs[f'{co}-{report_types[report_type]}'] = search_files(folder, self.terms, norm=True, norm_type=self.norm_type, count_type=self.count_type, year_index=True)
                        if save_csv:
                            dfs[f'{co}-{report_types[report_type]}'].to_csv(normed_f(co, report_type))
                    else:
                        dfs[f'{co}-{report_types[report_type]}'] = pd.read_csv(normed_f(co, report_type), index_col=0)

                    if reprocess or not os.path.exists(unnormed_f(co, report_type)):
                        dfs_unnormed[f'{co}-{report_types[report_type]}'] = search_files(folder, self.terms, norm=False, norm_type=None, count_type=self.count_type, year_index=True)
                        if save_csv:
                            dfs_unnormed[f'{co}-{report_types[report_type]}'].to_csv(unnormed_f(co, report_type))
                    else:
                        dfs_unnormed[f'{co}-{report_types[report_type]}'] = pd.read_csv(unnormed_f(co, report_type), index_col=0)

            return dfs, dfs_unnormed
    
        def _log(self, message:str):
            logger.debug(f'{self.uuid}: {message}')


    return Term, TermCollection


@app.cell
def _(Term, TermCollection, plt):
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

    TERMS = TermCollection(
        [
              Term('Solar', SOLAR_REGEX, 
                   'gold'),
              Term('Wind', WIND_REGEX, 
                   'powderblue'),
              Term('Hydrogen (green)', GREEN_HYDROGEN_REGEX, 
                   'turquoise', dash=True, edgecolour='chartreuse', hatch='....'),
              Term('Hydrogen (blue)', BLUE_HYDROGEN_REGEX, 
                   'turquoise', dash=True, edgecolour='blue', hatch='....'),
              Term('Hydrogen', HYDROGEN_REGEX, 
                   'turquoise'),
              Term('CCS', CCS_REGEX, 
                   'dimgray'),
              Term('Biofuel', BIOFUEL_REGEX, 
                   'tab:brown', dash=True, edgecolour='peru', hatch='....'),
              Term('Biogas', BIOGAS_REGEX, 
                   'tab:brown', dash=True, edgecolour='lightgrey', hatch='....'),
              Term('Geothermal', GEOTHERMAL_REGEX, 
                   'tab:red'),
              Term('Tidal', TIDAL_REGEX, 
                   'steelblue'),

              Term('Storage', STORAGE_REGEX, 
                   'pink'),
              Term('EVs', EVS_REGEX, 
                   'lightgrey'),
              Term('Nature', NATURE_REGEX, 
                   'darkgreen'),
              Term('Efficiency', EFFICIENCY_REGEX, 
                   'fuchsia'),

              Term('Nuclear (conventional)', NUCLEAR_CONVENTIONAL_REGEX, 
                   'tab:olive'),
              Term('Nuclear (SMR)', NUCLEAR_SMR_REGEX, 
                   'tab:olive', dash=True, edgecolour='tab:grey', hatch='....'),
              Term('Nuclear fusion', NUCLEAR_FUSION_REGEX, 
                   'tab:olive', dash=True, edgecolour='gold', hatch='....'),
         ],
         folder_name='all-terms'
    )

    TERMS.plot_colours()
    plt.show()
    return (
        BIOFUEL_REGEX,
        BIOGAS_REGEX,
        BLUE_HYDROGEN_REGEX,
        CCS_REGEX,
        EFFICIENCY_REGEX,
        EVS_REGEX,
        GEOTHERMAL_REGEX,
        GREEN_HYDROGEN_REGEX,
        HYDROGEN_REGEX,
        NATURE_REGEX,
        NUCLEAR_CONVENTIONAL_REGEX,
        NUCLEAR_FUSION_REGEX,
        NUCLEAR_SMR_REGEX,
        SOLAR_REGEX,
        STORAGE_REGEX,
        TERMS,
        TIDAL_REGEX,
        WIND_REGEX,
    )


@app.cell
def _(
    BIOFUEL_REGEX,
    BIOGAS_REGEX,
    BLUE_HYDROGEN_REGEX,
    CCS_REGEX,
    EFFICIENCY_REGEX,
    EVS_REGEX,
    GEOTHERMAL_REGEX,
    GREEN_HYDROGEN_REGEX,
    HYDROGEN_REGEX,
    NATURE_REGEX,
    NUCLEAR_CONVENTIONAL_REGEX,
    NUCLEAR_FUSION_REGEX,
    NUCLEAR_SMR_REGEX,
    SOLAR_REGEX,
    STORAGE_REGEX,
    TIDAL_REGEX,
    Term,
    TermCollection,
    WIND_REGEX,
    plt,
):
    TERMS_SUMMARY = TermCollection([
        Term('Solar', SOLAR_REGEX, 
             'gold'),
        Term('Wind', WIND_REGEX, 
             'powderblue'),
        Term('Hydrogen', f'{GREEN_HYDROGEN_REGEX}|{BLUE_HYDROGEN_REGEX}|{HYDROGEN_REGEX}', 
             'turquoise'),
        Term('CCS', CCS_REGEX, 
             'dimgray'),
        Term('Bioenergy', f'{BIOFUEL_REGEX}|{BIOGAS_REGEX}', 
             'tab:brown'),
        Term('Geothermal', GEOTHERMAL_REGEX, 
             'tab:red'),
        Term('Tidal', TIDAL_REGEX, 
             'steelblue'),

        Term('Storage', STORAGE_REGEX, 
             'pink'),
        Term('EVs', EVS_REGEX, 
             'lightgrey'),
        Term('Nature', NATURE_REGEX, 
             'darkgreen'),
        Term('Efficiency', EFFICIENCY_REGEX, 
             'fuchsia'),

        Term('Nuclear', f'{NUCLEAR_CONVENTIONAL_REGEX}|{NUCLEAR_SMR_REGEX}', 
             'tab:olive'),
        Term('Fusion', NUCLEAR_FUSION_REGEX, 
             'tab:olive', dash=True, edgecolour='gold', hatch='....'),
    ], folder_name='summary-terms')

    TERMS_SUMMARY.plot_colours()
    plt.show()
    return (TERMS_SUMMARY,)


@app.cell
def _(dataclass, os):
    @dataclass
    class Company:
        foldername: str
        alias: str

        @property
        def fpaths(self) -> dict[str, dict[str, str]]:
            report_types = ('an', 'sr')
            return {'normed': {report_type: os.path.join('out', 'mentions-normed', f'{self.foldername}-{report_type}.csv') for report_type in report_types},
                    'unnormed': {report_type: os.path.join('out', 'mentions', f'{self.foldername}-{report_type}.csv') for report_type in report_types}}

    company_aliases = {
            'shell': 'Shell',
            'bp': 'BP',
            'exxon': 'ExxonMobil',
            'chevron': 'Chevron',
            'total': 'TotalEnergies'
        }

    COS = [Company(co, alias) for co, alias in company_aliases.items()]
    return COS, company_aliases


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. Functions
    """)
    return


@app.cell
def _(Iterable, TERMS, Term, pymupdf, re):
    def search_file(
        file:str|pymupdf.Document, 
        terms: Iterable[Term] = TERMS, 
        *, 
        count_type: str = 'all', norm_type: str = 'words', norm:bool = True
    ) -> dict[str, int] | dict[str, int | float]:

        if isinstance(file, str): 
                doc = pymupdf.open(file) 
        elif isinstance(file, pymupdf.Document): 
                doc = file
        else:
            raise TypeError(f'file must be a filepath or Document instance. Got {type(file)}.')

        out= {term.alias: 0 for term in terms}

        n_words = 0

        for page in doc:
            txt = page.get_text().lower()

            n_words += len(re.findall(r"[^\W_]+", txt, re.M))

            for term in terms:
                matches = re.findall(term.regex, txt, re.I)
                if count_type == 'pages':
                    out[term.alias] += 1
                else:
                    out[term.alias] += len(matches)

        n_pages = len(doc)

        if norm:
            if norm_type.lower() == 'pages':
                out = {term: out[term]/n_pages for term in out}
            elif norm_type.lower() == 'words':
                out = {term: out[term]/n_words for term in out}
            else:
                raise ValueError(f'Invalid norm_type: {norm_type}. Must be "pages" or "words".')

        out['n_words'] = n_words
        out['n_pages'] = n_pages

        return out

    search_file('reports/Shell/Annual/shell-annual-report-2025.pdf', norm_type='words', norm = False)
    return (search_file,)


@app.cell
def _(Iterable, TERMS, Term, os, pd, re, search_file):
    def extract_year(filename: str) -> int:
        match = re.findall(r"\d{4}", filename)
        n_matches = len(match)
        if n_matches == 0:
            raise ValueError(f'filename does not contain a 4-digit year: {filename}')
        elif n_matches > 1:
            raise ValueError(f'more than 1 4-digit number found in filename: {filename}')
        else:
            return int(match[0])

    def search_files(
        folder:str, 
        terms: Iterable[Term] = TERMS, 
        *,
        count_type:str = 'all', norm: bool = True,
        year_index: bool = True,
        norm_type: str = 'words'
    ) -> pd.DataFrame:
        if not os.path.isdir(folder):
            raise ValueError('folder is not a folder!')

        out = {}
        for file in os.listdir(folder):
            if file.endswith('.pdf'):
                out[file] = search_file(os.path.join(folder, file), terms, count_type=count_type, norm=norm, norm_type=norm_type)
                out[file]['year'] = extract_year(file)

        out = pd.DataFrame(out).T
        out.index.name = 'filename'
        if year_index: out = out.reset_index().set_index('year')
        return out.sort_index()


    return (search_files,)


@app.cell
def _(COS, Iterable, TERMS, Term, company_aliases, mpl, np, pd, plt):
    def plot_mentions(
        df: pd.DataFrame, 
        *, 
        ax: plt.Axes | None = None, figsize: tuple[float]|tuple[int]|None = None,
        marker: str|None = None,
        plot_n: str|None = 'words', normed_by: str|None = 'words',
        terms: Iterable[Term] = TERMS
    ):

        if ax is None: _, ax = plt.subplots(figsize=figsize)

        for term in terms:
            ax.plot(df.index, df[term.alias], marker = marker, linestyle = term.dash_ls if term.dash else None, color = term.colour, gapcolor=term.edgecolour)

        if normed_by is not None:
            ax.set_ylabel(f'Term count/No. {normed_by}')
        else:
            ax.set_ylabel('Term count')
        ax.set_xticks(df.index)
        ax.set_xticklabels(df.index, rotation = 45)

        if not plot_n.startswith('n_'): plot_n = 'n_' + plot_n
        if plot_n in df:
            ax_n = ax.twinx()
            df[plot_n].plot(linestyle = 'None', marker = '2', color = 'k', alpha = 0.9, ax = ax_n)
            ax_n.set_ylabel(f'No. {plot_n.replace("n_", "")}')
            return ax, ax_n
        return ax

    def plot_rankings(
            df: pd.DataFrame,
            *,
            ax: plt.Axes | None = None,
            figsize: tuple[float]|tuple[int]|None = None,
            terms: Iterable[Term] = TERMS,
            lines: bool = True,
            zero_fill: str = 'nan'  # 'nan' or 'min' (fills with min rank + 1). ignored otherwise
    ):
        if ax is None: _, ax = plt.subplots(figsize=figsize)

        df_ranked = df.loc[:, [term.alias for term in TERMS]]
        if zero_fill.lower() == 'nan':
            df_ranked = df_ranked.replace(0, np.nan)
        elif zero_fill.lower() == 'min':
            min_rank = df_ranked[df_ranked > 0].min().min()
            df_ranked = df_ranked.replace(0, min_rank + 1)

        df_ranked = df_ranked.rank(ascending=False, method='min', axis=1)

        for term in terms:
            if lines:
                ax.plot(df_ranked.index, df_ranked[term.alias], 
                           marker = None, linestyle = term.dash_ls if term.dash else None, 
                           color = term.colour, gapcolor=term.edgecolour, zorder = 1)
            ax.scatter(df_ranked.index, df_ranked[term.alias], 
                            marker = term.marker, linestyle = 'None', 
                            color = term.colour, edgecolor=term.edgecolour, hatch=term.hatch)

        ax.invert_yaxis()
        ax.set_ylabel('Rank')
        ax.set_xticks(df_ranked.index)
        ax.set_xticklabels(df_ranked.index, rotation = 45)

        return ax

    def plot_companies(
        dfs: dict[str, pd.DataFrame], 
        terms: Iterable[Term] = TERMS, *,
        cos: Iterable[str] = COS, 
        rank: bool = False,
        rank_zero_fill: str = 'nan', rank_lines: bool = True,
        plot_n: str|None = 'words', normed_by: str|None = 'words',
        suptitle: str|None = None
    ):

        years_an = sorted({int(y) for co in cos for y in dfs[co + '-an'].index})
        years_sr = sorted({int(y) for co in cos for y in dfs[co + '-sr'].index})

        xmin_an, xmax_an = years_an[0], years_an[-1]
        xmin_sr, xmax_sr = years_sr[0], years_sr[-1]


        fig, axs = plt.subplots(len(cos), 2, figsize = (12, 3*len(cos) + 1))
        for i, co in enumerate(cos):
            axs1 = axs[i][0]
            axs2 = axs[i][1]
            if rank:
                plot_rankings(dfs[co+'-an'], ax = axs1, terms = terms, lines=rank_lines, zero_fill=rank_zero_fill)
                plot_rankings(dfs[co+'-sr'], ax = axs2, terms = terms, lines=rank_lines, zero_fill=rank_zero_fill)
            else:
                plot_mentions(dfs[co+'-an'], ax = axs1, terms = terms, plot_n=plot_n, normed_by=normed_by)
                plot_mentions(dfs[co+'-sr'], ax = axs2, terms = terms, plot_n=plot_n, normed_by=normed_by)

            axs1.set_xlim(xmin_an, xmax_an)
            axs2.set_xlim(xmin_sr, xmax_sr)

            if xmin_an < dfs[co+'-an'].index.min(): axs1.axvspan(xmin_an, dfs[co+'-an'].index.min(), color='lightgrey', alpha=0.5, zorder=-1)
            if xmax_an > dfs[co+'-an'].index.max(): axs1.axvspan(dfs[co+'-an'].index.max(), xmax_an, color='lightgrey', alpha=0.5, zorder=-1)
            if xmin_sr < dfs[co+'-sr'].index.min(): axs2.axvspan(xmin_sr, dfs[co+'-sr'].index.min(), color='lightgrey', alpha=0.5, zorder=-1)
            if xmax_sr > dfs[co+'-sr'].index.max(): axs2.axvspan(dfs[co+'-sr'].index.max(), xmax_sr, color='lightgrey', alpha=0.5, zorder=-1)

            if i == 0:
                axs1.set_title('Annual Reports')
                axs2.set_title('Sustainability Reports')

            if i < len(cos) - 1:
                axs1.set_xticks([])
                axs2.set_xticks([])
                axs1.set_xticklabels([])
                axs2.set_xticklabels([])
            else:
                axs1.set_xticks(np.arange(xmin_an, xmax_an + 1, 2))
                axs2.set_xticks(np.arange(xmin_sr, xmax_sr + 1, 2))
                axs1.set_xticklabels(np.arange(xmin_an, xmax_an + 1, 2))
                axs2.set_xticklabels(np.arange(xmin_sr, xmax_sr + 1, 2))
                axs1.set_xlabel('Year')
                axs2.set_xlabel('Year')


        # construct legend manually
        fig.legend(
            handles = [plt.Line2D([0], [0], color = term.colour, marker = None, linestyle = term.dash_ls if term.dash else None, label = term.alias, gapcolor=term.edgecolour) for term in terms],
            loc = 'lower center', ncol = (len(terms) + 1)//2, frameon = False, 
            bbox_to_anchor = (0.5, -0.03),
            alignment = 'center'
        )
        extra_labels = [mpl.patches.Patch(color='lightgrey', alpha=0.5, label='No data')]
        if plot_n and 'n_' + plot_n in dfs[cos[0] + '-an'] and not rank:
            extra_labels.append(plt.Line2D([0], [0], color = 'k', marker = '2', linestyle = 'None', label = f'Total {plot_n}'))
        fig.legend(
            handles = extra_labels,
            loc = 'lower center', ncol = len(extra_labels), frameon = False, 
            bbox_to_anchor = (0.5, -0.045),
        )

        if suptitle:
            fig.suptitle(suptitle, fontsize = 18, fontweight='light', y = 1.01)

        fig.tight_layout(pad = 0.75)

        for i, co in enumerate(cos):
            axs1 = axs[i][0]

            col_bot = axs1.get_position().y0
            col_top = axs1.get_position().y1
            if rank:
                offset = 0.055
            else:
                offset = 0.09 if normed_by else 0.065
            x = axs1.get_position().x0 - offset
            y = axs1.get_position().y0 + (col_top - col_bot) / 2
            fig.text(x, y, company_aliases.get(co, co.capitalize()), ha='center', va='center', fontweight='light', fontsize = 16, rotation = 90,
                     color = 'dimgrey')

    return (plot_companies,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2. Pipeline
    """)
    return


@app.cell
def _(COS, Iterable, TERMS_SUMMARY, Term, os, pd, search_files):
    def run_pipeline(terms: Iterable[Term], *, 
                     count_type: str = 'all', norm_type: str = 'words',
                     reprocess: bool = False, save_csv: bool = False):

        report_types = {'annual': 'an', 'sustainability-report': 'sr'}

        dfs = {}
        dfs_unnormed = {}

        if not os.path.exists('out/mentions'):
            os.makedirs('out/mentions')
        if not os.path.exists('out/mentions-unnormed'):
            os.makedirs('out/mentions-unnormed')

        normed_f = lambda co, rt: f'out/mentions/{co}-{report_types[rt]}.csv'
        unnormed_f = lambda co, rt: f'out/mentions-unnormed/{co}-{report_types[rt]}.csv'

        for co in COS:
            for report_type in report_types:
                folder = os.path.join('reports', co, report_type)

                if reprocess or not os.path.exists(normed_f(co, report_type)):
                    dfs[f'{co}-{report_types[report_type]}'] = search_files(folder, terms, norm=True, norm_type=norm_type, count_type=count_type, year_index=True)
                    if save_csv:
                        dfs[f'{co}-{report_types[report_type]}'].to_csv(normed_f(co, report_type))
                else:
                    dfs[f'{co}-{report_types[report_type]}'] = pd.read_csv(normed_f(co, report_type), index_col=0)

                if reprocess or not os.path.exists(unnormed_f(co, report_type)):
                    dfs_unnormed[f'{co}-{report_types[report_type]}'] = search_files(folder, terms, norm=False, norm_type=None, count_type=count_type, year_index=True)
                    if save_csv:
                        dfs_unnormed[f'{co}-{report_types[report_type]}'].to_csv(unnormed_f(co, report_type))
                else:
                    dfs_unnormed[f'{co}-{report_types[report_type]}'] = pd.read_csv(unnormed_f(co, report_type), index_col=0)

        return dfs, dfs_unnormed

    dfs, dfs_unnormed = run_pipeline(TERMS_SUMMARY, reprocess = True, save_csv = False)
    return dfs, dfs_unnormed


@app.cell
def _(dfs, plot_companies, plt):
    plot_companies(dfs, suptitle='Technology mentions - normalised by word count')

    plt.savefig('out/company_mentions.png', bbox_inches='tight', dpi=300)
    plt.show()
    return


@app.cell
def _(COS, dfs_unnormed, plot_companies, plt):
    plot_companies(dfs_unnormed, COS, normed_by=None,
                   suptitle='Technology mentions - unnormalised')
    plt.savefig('out/company_mentions_unnormed.png', bbox_inches='tight', dpi=300)
    plt.show()
    return


@app.cell
def _(COS, dfs_unnormed, plot_companies, plt):
    plot_companies(dfs_unnormed, COS, rank=True,
                   suptitle='Technology mention rankings')
    plt.savefig('out/company_mentions_rankings.png', bbox_inches='tight', dpi=300)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
