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
    if tab == 'xgboost':
        for i in range(1): #self.averaging.value()):
            model = XGBClassifier(
                n_estimators=xgb_n_estimators.value, 
                #subsample=self.subsample.value(), 
                random_state=i,
                #max_depth=self.max_depth.value()
            )
            print('X =', X)
            print('y =', y)
            model.fit(X, y)
            print('X1 =', X1[:10])
            preds = model.predict_proba(X1)[:,0].reshape(xgrid.shape)
            all_preds.append(preds)
    elif tab == 'catboost':
        for i in range(1):#self.cb_averaging.value()):
            model = CatBoostClassifier(
                num_trees=cat_num_trees.value, 
                random_seed=i,
            )
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
p = figure(title='tap = blue dot, shift-tap = orange dot',
           tools=TOOLS, width=300, height=300,
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

xgb_n_estimators = bm.Slider(start=1, end=100, value=10, title="n_estimators")
cat_num_trees = bm.Slider(start=1, end=100, value=10, title="num_trees")
xgb_n_estimators.on_change('value', slider_callback)
cat_num_trees.on_change('value', slider_callback)

xgb_tab = bm.TabPanel(child=xgb_n_estimators, title="XGBoost", name='xgboost')
cat_tab = bm.TabPanel(child=cat_num_trees, title="CatBoost", name='catboost')

cat1x, cat1y = [], []
cat2x, cat2y = [], []

def callback(event):
    if not event.modifiers['shift']:
        cat1x.append(event.x) 
        cat1y.append(event.y) 
        source1.data = dict(x=cat1x, y=cat1y)
    else:
        cat2x.append(event.x) 
        cat2y.append(event.y) 
        source2.data = dict(x=cat2x, y=cat2y)
    update()

p.on_event(Tap, callback)

def panelActive(attr, old, new):
    name = tabs.tabs[new].name
    update(name)
#    if name == 'xbgoost':
#
    print("the active panel is " + str(tabs.active))

tabs = bm.Tabs(tabs=[xgb_tab, cat_tab])
tabs.on_change('active', panelActive)

curdoc().add_root(bl.Column(
    p,
    tabs,
))
