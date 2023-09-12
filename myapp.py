from collections import defaultdict
from math import exp

import numpy as np
from bokeh.plotting import figure
import bokeh.models as bm
from bokeh.models import ColumnDataSource, Column
from bokeh.io import curdoc
from bokeh.events import Tap
import bokeh.layouts as bl
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

BLUE = "#1f77b4"
GREEN = "#2ca02c"
ORANGE = '#ff7f0e'
RED = '#d62728'
BLACK = '#000000'

def update(tab=None):
    X = np.vstack([
            np.column_stack([cat1x, cat1y]),
            np.column_stack([cat2x, cat2y]), 
    ])
    y = np.r_[np.zeros_like(cat1x), np.ones_like(cat2x)]
    xgrid, ygrid = np.meshgrid(np.linspace(0, 100, 101), np.linspace(0, 100, 101))
    X1 = np.column_stack([xgrid.ravel(), ygrid.ravel()])
    all_preds = []
#    tab = self.tabs.currentWidget().objectName()
    if tab is None:
        tab = tabs.tabs[tabs.active].name
    if tab == 'xgb':
        for i in range(params[tab]['averaging'].value):
            kwargs = {k: v.value for k, v in params[tab].items() if k != 'averaging'}
            for name in 'alpha', 'lambda', 'gamma':
                kwargs[name] = exp(kwargs[name])
            kwargs['random_state'] += i
            model = XGBClassifier(**kwargs)
#                n_estimators=xgb_n_estimators.value, 
#                #subsample=self.subsample.value(), 
#                random_state=xgb_random_state.value + i,
#                #max_depth=self.max_depth.value()
#            )
            print(f'update {i}')
#            print('X =', X)
#            print('y =', y)
            model.fit(X, y)
#            print('X1 =', X1[:10])
            preds = model.predict_proba(X1)[:,0].reshape(xgrid.shape)
            all_preds.append(preds)
    elif tab == 'cat':
        for i in range(params[tab]['averaging'].value):
            kwargs = {k: v.value for k, v in params[tab].items() if k != 'averaging'}
            kwargs['random_seed'] += i
            model = CatBoostClassifier(**kwargs)
#                num_trees=cat_num_trees.value, 
#                depth=cat_depth.value, 
#                random_seed=cat_random_seed.value + i,
#            )
            model.fit(X, y)
            preds = model.predict_proba(X1)[:,0].reshape(xgrid.shape)
            all_preds.append(preds)
#    elif tab == 'lightgbm':
#        for i in range(self.lg_averaging.value()):
#            model = LGBMClassifier(
#                    n_estimators=self.lg_n_estimators.value(), 
#                    min_data_in_leaf=3,
#                    random_state=i,
#            )
#            model.fit(X, y)
#            preds = model.predict_proba(X1)[:,0].reshape(xgrid.shape)
#            all_preds.append(preds)
    if len(all_preds) > 1:
        all_preds = np.stack(all_preds).mean(axis=0)
    else:
        all_preds = all_preds[0]
    source0.data['image'] = [all_preds]
    print('done')

TOOLS = "tap"
p = figure(title='tap=blue dot, shift-tap=orange dot; min 8 dots',
           tools=TOOLS, width=350, height=350,
           x_range=(0, 100), y_range=(0, 100))

source1 = ColumnDataSource(data=dict(x=[], y=[]))   
p.circle(source=source1, x='x', y='y') 

source2 = ColumnDataSource(data=dict(x=[], y=[]))   
p.circle(source=source2, x='x', y='y', line_color=ORANGE, fill_color=ORANGE)

image = np.zeros((101, 101))
source0 = ColumnDataSource(data={'image':[image]})
im = p.image('image', source=source0, x=0, y=0, dw=100, dh=100, level="image")#, palette="Spectral11")  

def slider_callback(attr, old, new):
    update()

params_dict = {
    'xgb': {
        'n_estimators': {'start': 1, 'end': 100, 'value': 10},
        'subsample':    {'start': 0.,'end': 1.,  'value': 1., 'step': 0.1},
        'alpha':        {'start': -16, 'end': 2,  'value': -16},
        'lambda':       {'start': -16, 'end': 2,  'value': 0},
        'gamma':        {'start': -16, 'end': 2,  'value': -16},
        'random_state': {'start': 1, 'end': 100, 'value': 10},
        'averaging':    {'start': 1, 'end': 10,  'value': 1},
    },
    'cat': {
        'num_trees':    {'start': 1, 'end': 100, 'value': 10},
        'l2_leaf_reg':  {'start': 1, 'end': 10,  'value': 3},
        'random_seed':  {'start': 1, 'end': 100, 'value': 10},
        'averaging':    {'start': 1, 'end': 10,  'value': 1},
    }
}
        
params = defaultdict(dict)
tab_contents = {}
for model, pp in params_dict.items():
    for name, kwargs in pp.items():
        kwargs['title'] = name
        params[model][name] = bm.Slider(**kwargs)
        params[model][name].on_change('value', slider_callback)
    tab_contents[model] = bl.Column(*list(params[model].values()))

# _____________ tabs _______________ 
xgb_tab = bm.TabPanel(child=tab_contents['xgb'], title="XGBoost", name='xgb')
cat_tab = bm.TabPanel(child=tab_contents['cat'], title="CatBoost", name='cat')

cat1x, cat1y = [], []
cat2x, cat2y = [], []

def on_tap(event):
    class_id = active_class
    if event.modifiers['shift']:
        class_id = 3 - class_id
    if class_id == 1:
        cat1x.append(event.x) 
        cat1y.append(event.y) 
        source1.data = dict(x=cat1x, y=cat1y)
    elif class_id == 2:
        cat2x.append(event.x) 
        cat2y.append(event.y) 
        source2.data = dict(x=cat2x, y=cat2y)
    update()

p.on_event(Tap, on_tap)

def panelActive(attr, old, new):
    name = tabs.tabs[new].name
    update(name)
#    if name == 'xbgoost':
#
    print("the active panel is " + str(tabs.active))

active_class = 1

def select_class(class_id):
    def wrapped(new):
        print(f'{class_id=}')
        global active_class
        if class_id == 1:
            orange.update(active=not new)
        elif class_id == 2:
            blue.update(active=not new)
        active_class = class_id
        if new is False:
            active_class = 3-active_class
    return wrapped

def on_reset():
    cat1x.clear()
    cat1y.clear()
    cat2x.clear()
    cat2y.clear()
    source1.data = dict(x=cat1x, y=cat1y)
    source2.data = dict(x=cat2x, y=cat2y)
    blue.update(active=True)
    update()

# __________ class buttons _______________
blue = bm.Toggle(label='Class 1', active=True)
orange = bm.Toggle(label='Class 2')
blue.on_click(select_class(1))
orange.on_click(select_class(2))

reset = bm.Button(label='Reset')
reset.on_click(on_reset)

tabs = bm.Tabs(tabs=[xgb_tab, cat_tab])
tabs.on_change('active', panelActive)

curdoc().add_root(bl.Column(
    p,
    bl.Row(blue, orange, reset),
    tabs,
))
