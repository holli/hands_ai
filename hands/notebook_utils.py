from hands.basics import *
from hands.data import *

_ = torch.manual_seed(42); np.random.seed(42); random.seed(42)


def model_requires_grad_info(model, print_all=True):
    #print("Layer: param.requires_grad")
    first = True
    for name, param in model.named_parameters():
        if '.bn.' in name: continue
        if first:
            print(f"{name}: {param.requires_grad}\n...")
            req = param.requires_grad
            first = False
            continue
        if param.requires_grad != req:
            print(f"{name}: {param.requires_grad}\n...")
            req = param.requires_grad
    print(f"{name}: {param.requires_grad}")


def model_load_backbone(model, h5_path, print_infos=True):
    state_old = model.state_dict()
    keys = list(state_old.keys())
    state_loaded = torch.load(h5_path)
    state = type(state_old)()

    skipped_layers = []
    for k_new in list(state_loaded.keys()):
        k_strip = k_new #.replace('backbone.', '')

        if k_strip not in keys or (state_old[k_strip].shape != state_loaded[k_new].shape):
            skipped_layers.append(k_strip)
            # del state_new[k_new]
        else:
            state[k_strip] = state_loaded[k_new]

    if len(state.keys()) < 100: warnings.warn("Loading only few items, check that its correct and ignore")

    model.load_state_dict(state, strict=False)
    loaded_layers = list(state.keys())
    if print_infos:
        print(f"Loaded to: {loaded_layers[0]} -> {loaded_layers[-1]}.\nSkipped from: {skipped_layers[0]}...")
    return list(state.keys()), skipped_layers


# examples
# plot_results(*list(zip(*wrong_samples[i:i+j])))
# examples
# plot_results(*list(zip(*wrong_samples[i:i+j])))
def plot_results(img_tensors, results_all, targets=None, infos=None, idxs=None, plt_kwargs=None, base_size=7):
    if not plt_kwargs:
        ncols = 2
        nrows = math.ceil(len(img_tensors)/ncols)
#         plt_kwargs = {'nrows':nrows, 'ncols':ncols, 'figsize':(14, nrows*7)}
        plt_kwargs = {'nrows':nrows, 'ncols':ncols, 'figsize':(base_size*2, nrows*base_size)}
        if len(img_tensors) == 1:
            plt_kwargs['ncols']=1
            plt_kwargs['figsize']=(8, 8)

    _, axes = plt.subplots(squeeze=False, **plt_kwargs)
    colors = ['red', 'blue', 'green', 'yellow']

    for b, ax in enumerate(axes.flatten()):
        if b+1 > len(img_tensors): break

        title = None
        if infos is not None: title = str(infos[b])
        if idxs is not None: title += " " + str(idxs[b])
        if title is not None: ax.set_title(title)

        img = img_tensors[b]
        _ = ax.imshow(fastai.vision.image.image2np(img))

        for l_i, x, y, dx, dy, obj_p, l_p in results_all[b]:
            x, y = x*img.shape[2], y*img.shape[1]
            color = colors[int(l_i)%len(colors)]
            l = HandItems.CLASS_NAMES[int(l_i)]
            ax.add_patch(patches.Circle((x, y), 2, fill=True, edgecolor=color, lw=1))

            arrow_size = 100
            ax.add_patch(patches.Arrow(x, y, dx*arrow_size, dy*arrow_size, color=color, lw=1))

            ly = y+20
            lx = max(5, x-30)
            label = HandItems.CLASS_NAMES[l_i]
            patch = ax.text(lx, ly, label, verticalalignment='top', color=color, fontsize=16, weight='normal')
            patch.set_path_effects([patheffects.Stroke(linewidth=1, foreground='black', alpha=0.5), patheffects.Normal()])

            label_add = "{:.2f} -> {:.2f}".format(obj_p, l_p)
            patch = ax.text(lx, ly+20, label_add, verticalalignment='top', color=color, fontsize=16, weight='normal')
            patch.set_path_effects([patheffects.Stroke(linewidth=1, foreground='black', alpha=0.5), patheffects.Normal()])

        if targets is not None:
            for l_i, y, x, dy, dx in targets[b]:
                if l_i > 0:
                    x, y = x*img.shape[2], y*img.shape[1]
                    color = colors[int(l_i)%len(colors)]
                    l = HandItems.CLASS_NAMES[int(l_i)]
                    ax.add_patch(patches.Circle([x, y], 10, fill=False, edgecolor=color, lw=1))
                    if dx != -100:
                        arrow_size = 1200
                        ax.add_patch(patches.Arrow(x, y, dx*arrow_size, dy*arrow_size, color='white', alpha=1, lw=1))


def plot_hand_tensors(imgs, his, plt_kwargs=None):
    if not plt_kwargs: plt_kwargs = {'nrows':math.ceil(len(imgs)/3), 'ncols':3, 'figsize':(10, min(9, math.ceil(len(imgs)/3)*3))}
    _, axes = plt.subplots(**plt_kwargs)
    colors = np.array([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]])
    for i, ax in enumerate(axes.flat):
        if i > len(imgs)-1: break
        img = imgs[i]
        _ = ax.imshow(fastai.vision.image.image2np(img))
        for l, y, x, dy, dx in his[i]:
            if l > 0:
                color = colors[int(l)%len(colors)]
                x, y = x*img.shape[2], y*img.shape[1]
                l = HandItems.CLASS_NAMES[int(l)]
                patch = ax.add_patch(patches.Circle([x, y], 10, fill=False, edgecolor='blue', lw=2))
                patch = ax.text(x, y+20, l, verticalalignment='top', color=color, fontsize=16, weight='normal')
                patch.set_path_effects([patheffects.Stroke(linewidth=1, foreground='black', alpha=0.5), patheffects.Normal()])


def plot_outputs(outputs, img_tensors, targets=None, plt_kwargs=None):
    if not plt_kwargs:
        nrows = math.ceil(len(outputs)/2)
        plt_kwargs = {'nrows':nrows, 'ncols':2, 'figsize':(14, min(14, nrows*7))}
        if len(outputs) == 1:
            plt_kwargs['ncols']=1
            plt_kwargs['figsize']=(8, 8)

    _, axes = plt.subplots(squeeze=False, **plt_kwargs)
    colors = ['red', 'blue', 'green', 'yellow']
    outputs_s = outputs.sigmoid()

    for b, ax in enumerate(axes.flatten()):
        if b+1 > len(outputs): break

        img = img_tensors[b]
        if targets is not None: target = targets[b]
        else: target = None
        _ = ax.imshow(fastai.vision.image.image2np(img))

        grid_shape = outputs.shape[2]
        grid_sz = img.shape[1]/grid_shape
        for i in range(grid_shape):
            for j in range(grid_shape):
                xy_grid = (j*grid_sz, i*grid_sz)
                alpha = outputs_s[b, 4, i, j].item()*2
                patch = patches.Rectangle(xy_grid, grid_sz, grid_sz, fill=False, alpha=alpha, edgecolor='blue', lw=1)
                _ = ax.add_patch(patch)

                if alpha > .1:
                    l_sm = outputs[b,5:,i,j].softmax(0)
                    l_i = int(l_sm.argmax(0))
                    color = colors[int(l_i)%len(colors)]
                    label = l_i
                    label = HandItems.CLASS_NAMES[l_i]
                    # print(outputs_s[b, 0:2, i, j].cpu().detach().numpy() * grid_sz)
                    xy_cent = outputs_s[b, 0:2, i, j].cpu().detach().numpy()
                    xy_cent = xy_cent.T[[1, 0]] * grid_sz + xy_grid
                    _ = ax.add_patch(patches.Circle(xy_cent, 2, fill=False, edgecolor=color, lw=1))

                    xy_angle = outputs[b, 2:4, i, j].tanh().cpu().detach().numpy()
                    xy_angle = xy_angle.T[[1, 0]] * 50
                    #print(xy_angle)
                    _ = ax.add_patch(patches.Arrow(*xy_cent, *xy_angle, color=color, lw=1))

                    label = "{}, {:.2f} -> {:.2f}".format(label, alpha/2, l_sm[l_i])
                    patch = ax.text(xy_cent[0], xy_cent[1]+20, label, verticalalignment='top', color=color, fontsize=16, weight='normal')
                    patch.set_path_effects([patheffects.Stroke(linewidth=1, foreground='black', alpha=0.5), patheffects.Normal()])

        if target is not None:
            for l_i, y, x, dy, dx in target:
                if l_i > 0:
                    x, y = x*img.shape[2], y*img.shape[1]
                    color = colors[int(l_i)%len(colors)]
                    l = HandItems.CLASS_NAMES[int(l_i)]
                    patch = ax.add_patch(patches.Circle([x, y], 10, fill=False, edgecolor=color, lw=1))
