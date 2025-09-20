import numpy as np
import matplotlib.pyplot as plt

def test_image_circle_cross(N,M):
    X,Y = np.meshgrid(np.linspace(-1,1,N), np.linspace(-1,1,M), sparse=False)
    absorp_0=(abs(X)<0.601)*(abs(Y)<0.199)+(abs(X)<0.199)*(abs(Y)<0.601)
    absorp_0=absorp_0.astype('int')
    absorp_1=(X**2+Y**2<=0.501**2)*(X**2+Y**2>=0.45**2)
    absorp_1=absorp_1.astype('int')
    absorp_2=(abs(X)<0.3)*(abs(Y)<0.055)+(abs(X)<0.055)*(abs(Y)<0.3)
    absorp_2=absorp_2.astype('int')
    absorp_3=(abs(X)**25+abs(Y)**25<=0.601**25)*(abs(X)**25+abs(Y)**25>=0.551**25)
    absorp_3=absorp_3.astype('int')
    absorp=absorp_0+absorp_1+absorp_2+absorp_3
    phase = ((abs(X+Y) <=0.101)+(abs(X-Y) <= 0.101)).astype('int')
    support_mask=((abs(X)**2+abs(Y)**2)<=0.7).astype('int')   # constant circular bump
    return support_mask*(0.1*absorp+0.1*complex(0,1)*phase)

def test_image_cells():
    absorp=np.load('cell1.npy')
    phase=np.load('cell2.npy')
    N,M = absorp.shape
    X,Y = np.meshgrid(np.linspace(-1,1,N), np.linspace(-1,1,M), sparse=False)    
    support_mask=((abs(X)**2+abs(Y)**2)<=0.6).astype('int')                    # constant circular bump
    return (0.01+0.01*complex(0,1))*support_mask+(0.2*absorp + 0.2*complex(0,1) * phase)

def show_comparison_results(contrast, reco_intcorr,reco_meanint):
    vmin=0.
    vmax=0.2
    #cmap='Reds'
    fontsize=20
    levels=40
    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    # Plot each image

    axs[0, 0].imshow(contrast.real, vmin=vmin, vmax=vmax)
    axs[0, 0].set_title('exact',fontsize=fontsize)
    axs[0,0].set_ylabel("absorption",fontsize=fontsize)
    axs[0,0].set_xticks([])
    axs[0,0].set_yticks([])
    # axs[0, 0].axis('off')

    axs[1,0].imshow(contrast.imag, vmin=vmin, vmax=vmax)
    axs[1,0].set_ylabel("phase",fontsize=fontsize)
    axs[1,0].set_xticks([])
    axs[1,0].set_yticks([])
    #axs[1, 0].axis('off')

    axs[0, 1].imshow(reco_intcorr.real, vmin=vmin, vmax=vmax)
    axs[0, 1].set_title('intensity correlations',fontsize=fontsize)
    axs[0, 1].axis('off')

    axs[1, 1].imshow(reco_intcorr.imag, vmin=vmin, vmax=vmax)
    axs[1, 1].axis('off')

    im5 = axs[0, 2].imshow(reco_meanint.real, vmin=vmin, vmax=vmax)
    axs[0, 2].set_title('mean intensity',fontsize=fontsize)
    axs[0, 2].axis('off')
    fig.colorbar(im5, ax=axs[0, 2])

    im6 = axs[1, 2].imshow(reco_meanint.imag, vmin=vmin, vmax=vmax)
    axs[1, 2].axis('off')
    fig.colorbar(im6, ax=axs[1, 2])

    # Adjust layou
    plt.tight_layout()
    #plt.subplot_tool()
    plt.show()


    plt.figure(figsize=(14, 7))
    N=contrast.shape[0]/2
    plt.plot(reco_meanint[int(N/2), :].imag, label='Mean intensity')
    plt.plot(reco_intcorr[int(N/2), :].imag, label='Intensity correlations')
    plt.plot(contrast[int(N/2), :].imag, label='Exact phase')
    plt.legend()
    plt.show()

def show_results(contrast, reco, vmin=0.,vmax=0.2):
    #cmap='Reds'
    fontsize=10
    levels=40
    fig, axs = plt.subplots(2, 2, figsize=(14, 7))
    # Plot each image

    im1 = axs[0, 0].imshow(contrast.real, vmin=vmin, vmax=vmax)
    axs[0, 0].set_title('absorption',fontsize=fontsize)
    axs[0, 0].axis('off')

    im2 = axs[0, 1].imshow(reco.real, vmin=vmin, vmax=vmax)
    axs[0, 1].set_title('Recovered absorption',fontsize=fontsize)
    axs[0, 1].axis('off')
    fig.colorbar(im2, ax=axs[0, 1])

    im3 = axs[1, 0].imshow(contrast.imag, vmin=vmin, vmax=vmax)
    axs[1, 0].set_title('Exact phase',fontsize=fontsize)
    axs[1, 0].axis('off')
    fig.colorbar(im3, ax=axs[1, 0])

    im4 = axs[1, 1].imshow(reco.imag, vmin=vmin, vmax=vmax)
    axs[1, 1].set_title('Recovered phase',fontsize=fontsize)
    axs[1, 1].axis('off')
    fig.colorbar(im4, ax=axs[1, 1])
    
    # Adjust layout
    plt.tight_layout()
    #plt.subplot_tool()
    plt.show()

def show_comparison_results(contrast, reco_intcorr,reco_meanint):
    vmin=0.
    vmax=0.2
    #cmap='Reds'
    fontsize=10
    levels=40
    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    # Plot each image

    axs[0, 0].imshow(contrast.real, vmin=vmin, vmax=vmax)
    axs[0, 0].set_title('exact',fontsize=fontsize)
    axs[0,0].set_ylabel("absorption")
    axs[0, 0].axis('off')

    axs[1, 0].imshow(contrast.imag, vmin=vmin, vmax=vmax)
    axs[0,1].set_ylabel("phase")
    axs[1, 0].axis('off')

    axs[0, 1].imshow(reco_intcorr.real, vmin=vmin, vmax=vmax)
    axs[0, 1].set_title('intesity correlations',fontsize=fontsize)
    axs[0, 1].axis('off')

    axs[1, 1].imshow(reco_intcorr.imag, vmin=vmin, vmax=vmax)
    axs[1, 1].axis('off')

    im5 = axs[0, 2].imshow(reco_meanint.real, vmin=vmin, vmax=vmax)
    axs[0, 2].set_title('mean intensity',fontsize=fontsize)
    axs[0, 2].axis('off')
    fig.colorbar(im5, ax=axs[0, 2])

    im6 = axs[1, 2].imshow(reco_meanint.imag, vmin=vmin, vmax=vmax)
    axs[1, 2].axis('off')
    fig.colorbar(im6, ax=axs[1, 2])

    # Adjust layou
    plt.tight_layout()
    #plt.subplot_tool()
    plt.show()


    plt.figure(figsize=(14, 7))
    N=contrast.shape[0]/2
    plt.plot(reco_meanint[int(N/2), :].imag, label='Mean intensity')
    plt.plot(reco_intcorr[int(N/2), :].imag, label='Intensity correlations')
    plt.plot(contrast[int(N/2), :].imag, label='Exact phase')
    plt.legend()
    plt.show()

    plt.figure(figsize=(14, 7))
    plt.plot(reco_meanint[int(N/2), :].real, label='Mean intensity')
    plt.plot(reco_intcorr[int(N/2), :].real, label='Intensity correlations')
    plt.plot(contrast[int(N/2), :].real, label='Exact absorption')
    plt.legend()
    plt.show()