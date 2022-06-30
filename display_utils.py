import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def map_w2v_to_quadrant(input_val: float, input_aro: float, input_dom: float) -> float:
    ''' Maps dimensional values from wav2vec2 (0 -> 1) to quadrant (-1 -> 1) values. '''
    output_val = (input_val - 0.5) * 2
    output_aro = (input_aro - 0.5) * 2
    output_dom = (input_dom - 0.5) * 2
    return output_val, output_aro, output_dom

def quadrant_chart(input_val: float, input_aro: float, xtick_labels=None, ytick_labels=None, ax=None) -> None:
    '''
    Function to display dimensional emotion data on a quadrant chart
    Function modifed from code provided from: https://gist.githubusercontent.com/ctnormand1/d9326e1556ed199d92063b89675450ef/raw/a707f33883ff59e2f738d236d4c555de56346562/simple_quadrant_chart.py
    '''
    data = pd.DataFrame({'x': input_val, 'y': input_aro})

    # Create axes
    ax = plt.axes()

    # Set x and y limits
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

    ax.scatter(x=data['x'], y=data['y'], c=input_aro, cmap = 'Spectral', edgecolor='darkblue', s=200,
    zorder=99)
    ax.axvline(0, c='k', lw=1)
    ax.axhline(0, c='k', lw=1)

if __name__ == '__main__':
    
    quadrant_chart(
        [-1,-.707, 0, .707, 1, .707, -.707, 0, 0],
        [0, .707, 0, -.707, 0, .707, -.707, 1, -1],
        xtick_labels=['Low', 'High'],
        ytick_labels=['Low', 'High']
    )

    plt.title('Arousal vs Valence', fontsize=16)
    plt.ylabel('Arousal', fontsize=14)
    plt.xlabel('Valence', fontsize=14)
    plt.grid(True, animated=True, linestyle='--', alpha=0.5)
    plt.show()
    # plot_color_gradients('Diverging',
    #                  ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
    #                   'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'])
    # plt.show()