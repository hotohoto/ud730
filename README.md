# ud730

Personal workspace for [Deep Learning](https://classroom.udacity.com/courses/ud730) course on Udacity.

## Snippets

### Run jupyter notebook
```shell
jupyter notebook
```
1. Open `.ipynb` file.
1. Edit and run.

You may get error message like below.
```
"http://localhost:8888/tree?token=XXX" doesn’t understand the “open location” message.
```
Then, open `~/.jupyter/jupyter_notebook_config.py` and insert the line below at proper place.

```
c.NotebookApp.browser = u'chrome'
```

In case you don't have `~/.jupyter/jupyter_notebook_config.py` file, run this first.

```shell
jupyter notebook --generate-config
```

### Reference
- https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/udacity
