import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import librosa

def plot_spectrogram(spectral_map, 
                     time_values_present, 
                     frequencies_values_present, 
                     figname = "spectrogram", 
                     save = False,
                     title = None, 
                     savepath = None,
                     showcolorbar = True,
                     log_ = False, 
                     vmin = None, 
                     vmax = None,
                     width = 700,
                     height = 500):
    if log_:
        spectral_map = librosa.core.power_to_db(np.abs(spectral_map)**2, ref=1, amin=1e-10, top_db=80.0)
    else:
        spectral_map = np.abs(spectral_map)
    
    if not vmin:
        vmin = spectral_map.min()
    if not vmax:
        vmax = spectral_map.max()
    
    fig = go.Figure(data=go.Heatmap(z=spectral_map))
    fig.show()
    df = frequencies_values_present[1] - frequencies_values_present[0]
    dt = time_values_present[1] - time_values_present[0]
    im = axx.imshow(spectral_map, aspect = 'auto', origin = 'lower', extent = (time_values_present[0] - dt/2, time_values_present[-1] + dt/2, frequencies_values_present[0] - df/2, frequencies_values_present[-1] + df/2), vmin = vmin, vmax = vmax, cmap = cmap)
    fig.update_layout(autosize = False, height = height, width = width, 
                        margin=dict(l=5, r=5, t=5, b=5), xaxis_title='Frequency[Hz]',
                        yaxis_title='Cumulative Activity')
    
    if not showcolorbar:
        fig.update_traces(showscale = False)
    
    if title:
        fig.update_layout(title_text = title)
    
    if save:
        if savepath:
            figname = os.path.join(savepath, figname + ".jpg")
        else:
            figname += ".jpg"
        fig.write_image(figname, width = height + 50, height = width + 50)
    else:
        fig.show()
    del fig

def plot_3dplot(spectral_map, 
                time, frequency, 
                figname = "3dplot", 
                save = False,
                title = None, 
                savepath = None, 
                vmin = None, 
                vmax = None, 
                log_ = False,
                width = 700,
                height = 500,
                camera = None,
                showcolorbar = True,
                colorbar = "Viridis",
                fontsize = 16):
    if log_:
        spectral_map = librosa.core.power_to_db(np.abs(spectral_map)**2, ref=1, amin=1e-10, top_db=80.0)
    else:
        spectral_map = np.abs(spectral_map)
    
    if not vmin:
        vmin = spectral_map.min()
    if not vmax:
        vmax = spectral_map.max()
    
    fig = go.Figure(data = [go.Surface(z=spectral_map, x = time, y = frequency/1000, cmax = vmax, cmin = vmin, colorscale = colorbar)])
    if not camera:
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=-0.2),
            eye=dict(x=1.25*1.5, y=0.4*1.5, z=0.5*1.5)
        )
    
    fig.update_layout(autosize = False, height = height, width = width, 
                        margin=dict(l=0, r=0, t=0, b=0), scene_camera=camera, 
                        scene_xaxis_title='Time [s]',
                        scene_yaxis_title='Frequency [KHz]',
                        scene_zaxis_title='Amplitude',
                        scene_zaxis_range = [vmin, vmax], font_size = fontsize)
    
    if not showcolorbar:
        fig.update_traces(showscale = False)
    
    if title:
        fig.update_layout(title_text = title)
    
    if save:
        if savepath:
            figname = os.path.join(savepath, figname + ".jpg")
        else:
            figname += ".jpg"
        fig.write_image(figname, width = width + 50, height = height + 50)
    else:
        fig.show()
    del fig

def plot_cumulative_freq_response(spectral_map, 
                                time, frequency, 
                                figname = "cumulative_freq", 
                                save = False,
                                title = None, 
                                savepath = None, 
                                vmin = None, 
                                vmax = None, 
                                log_ = False,
                                width = 700,
                                height = 500):
    if log_:
        spectral_map = librosa.core.power_to_db(np.abs(spectral_map)**2, ref=1, amin=1e-10, top_db=80.0)
    else:
        spectral_map = np.abs(spectral_map)
    
    if not vmin:
        vmin = spectral_map.min()
    if not vmax:
        vmax = spectral_map.max()

    cumul_freq_resp_vector = np.squeeze(np.mean(spectral_map, axis = 1))

    fig = go.Figure(data=go.Scatter(x=frequency/1000, y=cumul_freq_resp_vector))
    fig.update_layout(autosize = False, height = height, width = width, 
                        margin=dict(l=5, r=5, t=5, b=5), xaxis_title='Frequency [KHz]',
                   yaxis_title='Cumulative Amplitude', yaxis_range=[vmin,vmax])
    
    if title:
        fig.update_layout(title_text = title)
    
    if save:
        if savepath:
            figname = os.path.join(savepath, figname + ".jpg")
        else:
            figname += ".jpg"
        fig.write_image(figname, width = height + 50, height = width + 50)
    else:
        fig.show()
    del fig
