/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the 'License');
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an 'AS IS' BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
import * as d3 from 'd3';

const FRAC_VIEWPOINT_AREA: number = 0.8;
export class Minimap {
  /** The minimap container. */
  private minimap: HTMLElement;
  /** The canvas used for drawing the mini version of the svg. */
  private canvas: HTMLCanvasElement;
  /** A buffer canvas used for temporary drawing to avoid flickering. */
  private canvasBuffer: HTMLCanvasElement;
  /** The minimap svg used for holding the viewpoint rectangle. */
  private minimapSvg: SVGSVGElement;
  /** The rectangle showing the current viewpoint. */
  private viewpoint: SVGRectElement;
  /**
   * The scale factor for the minimap. The factor is determined automatically
   * so that the minimap doesn't violate the maximum width/height specified
   * in the constructor. The minimap maintains the same aspect ratio as the
   * original svg.
   */
  private scaleMinimap: number = 1;
  /** The main svg element. */
  private svg: SVGSVGElement;
  /** The svg group used for panning and zooming the main svg. */
  private zoomG: SVGGElement;
  /** The zoom behavior of the main svg. */
  private mainZoom: d3.ZoomBehavior<any, any>;
  /** The maximum width and height for the minimap. */
  private maxWandH: number;
  /** The last translation vector used in the main svg. */
  private translate: [number, number] = [0, 0];
  /** The last scaling factor used in the main svg. */
  private scaleMain: number = 1;
  /** The coordinates of the viewpoint rectangle. */
  private viewpointCoord: {
    x: number;
    y: number;
  };

  /** The current size of the minimap */
  private minimapSize: {
    width: number;
    height: number;
  } = { width: 150, height: 150 };

  /** Padding (px) due to the main labels of the graph. */
  private labelPadding: number;
  /**
   * Constructs a new minimap.
   *
   * @param svg The main svg element.
   * @param zoomG The svg group used for panning and zooming the main svg.
   * @param mainZoom The main zoom behavior.
   * @param minimap The minimap container.
   * @param maxWandH The maximum width/height for the minimap.
   * @param labelPadding Padding in pixels due to the main graph labels.
   */
  constructor(
    svg: SVGSVGElement,
    zoomG: SVGGElement,
    mainZoom: d3.ZoomBehavior<any, any>,
    minimap: HTMLElement,
    maxWandH: number,
    labelPadding: number,
  ) {
    this.svg = svg;
    this.labelPadding = labelPadding;
    this.zoomG = zoomG;
    this.mainZoom = mainZoom;
    this.maxWandH = maxWandH;
    let $shadowRoot = d3.select(minimap.shadowRoot as unknown as Element);
    // The minimap will have 2 main components: the canvas showing the content
    // and an svg showing a rectangle of the currently zoomed/panned viewpoint.
    let $minimapSvg = $shadowRoot.select('svg');
    // Make the viewpoint rectangle draggable.
    let $viewpoint = $minimapSvg.select('rect');
    let dragmove = (event: d3.D3DragEvent<any, any, any>): void => {
      let width = Number($viewpoint.attr('width'));
      let height = Number($viewpoint.attr('height'));
      this.viewpointCoord.x = event.x- (width / 2);;
      this.viewpointCoord.y = event.y- (height / 2);;
      this.updateViewpoint();
    };
    this.viewpointCoord = { x: 0, y: 0 };
    let drag = d3.drag().subject(Object).on('drag', dragmove);
    $viewpoint.datum(this.viewpointCoord as any).call(drag as any);
    // Make the minimap clickable.
    $minimapSvg.on('click', (event: Event): void => {
      if (event.defaultPrevented) {
        // This click was part of a drag event, so suppress it.
        return;
      }
      // Update the coordinates of the viewpoint.
      let width = Number($viewpoint.attr('width'));
      let height = Number($viewpoint.attr('height'));
      let clickCoords = d3.pointer(event, $minimapSvg.node() as any);
      this.viewpointCoord.x = clickCoords[0] - (width / 2);
      this.viewpointCoord.y = clickCoords[1] - (height / 2);
      this.updateViewpoint();
    });
    this.viewpoint = <SVGRectElement>$viewpoint.node();
    this.minimapSvg = <SVGSVGElement>$minimapSvg.node();
    this.minimap = minimap;
    this.canvas = <HTMLCanvasElement>$shadowRoot.select('canvas.first').node();
    this.canvasBuffer = <HTMLCanvasElement>$shadowRoot.select('canvas.second').node();
    this.update();
  }

  /**
   * Redraws the minimap. Should be called whenever the main svg
   * was updated (e.g. when a node was expanded).
   */
  update(): void {
    let sceneSize: DOMRect | null = null;
    try {
      // Get the size of the entire scene.
      sceneSize = this.zoomG.getBBox();
      if (sceneSize.width === 0) {
        return;
      }
    } catch (e) {
      return;
    }
    let $svg = d3.select(this.svg);
    let stylesText = '';
    const anySvg = this.svg as any;
    const rootNode = anySvg.getRootNode ? anySvg.getRootNode() : this.svg.parentNode;
    const styleSheets = rootNode.styleSheets;
    for (let k = 0; k < styleSheets.length; k++) {
      try {
        let cssRules = (<any>styleSheets[k]).cssRules || (<any>styleSheets[k]).rules;
        if (cssRules === null) {
          continue;
        }
        for (let i = 0; i < cssRules.length; i++) {
          // Remove tf-* selectors from the styles.
          stylesText += `${cssRules[i].cssText.replace(/ ?tf-[\w-]+ ?/g, '')}\n`;
        }
      } catch (e: any) {
        if (e.name !== 'SecurityError') {
          throw e;
        }
      }
    }
    let svgStyle = $svg.append('style');
    svgStyle.text(stylesText);
    let $zoomG = d3.select(this.zoomG);
    let zoomTransform = $zoomG.attr('transform');
    $zoomG.attr('transform', null);
    sceneSize.height += sceneSize.y;
    sceneSize.width += sceneSize.x;
    sceneSize.height += this.labelPadding * 2;
    sceneSize.width += this.labelPadding * 2;
    $svg.attr('width', sceneSize.width).attr('height', sceneSize.height);
    this.scaleMinimap = this.maxWandH / Math.max(sceneSize.width, sceneSize.height);
    // canvas宽度缩小一半，图像填充满需要乘2
    this.minimapSize = {
      width: sceneSize.width * this.scaleMinimap * 2,
      height: sceneSize.height * this.scaleMinimap,
    };
    d3.select(this.minimapSvg).attr(this.minimapSize as any);
    d3.select(this.canvasBuffer).attr(this.minimapSize as any);
    if (this.translate != null && this.zoom != null) {
      requestAnimationFrame(() => this.zoom());
    }
    let svgXml = new XMLSerializer().serializeToString(this.svg);
    svgStyle.remove();
    $svg.attr('width', null).attr('height', null);
    $zoomG.attr('transform', zoomTransform);
    let image = new Image();
    image.onload = (): void => {
      // Draw the svg content onto the buffer canvas.
      let context = this.canvasBuffer.getContext('2d');
      context?.clearRect(0, 0, this.canvasBuffer.width, this.canvasBuffer.height);
      context?.drawImage(image, 0, 0, this.minimapSize.width, this.minimapSize.height);
      requestAnimationFrame(() => {
        // Hide the old canvas and show the new buffer canvas.
        d3.select(this.canvasBuffer).style('display', null);
        d3.select(this.canvas).style('display', 'none');
        // Swap the two canvases.
        [this.canvas, this.canvasBuffer] = [this.canvasBuffer, this.canvas];
      });
    };
    image.onerror = (): void => {
      let blob = new Blob([svgXml], { type: 'image/svg+xml;charset=utf-8' });
      image.src = (URL as any).createObjectURL(blob);
    };
    image.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svgXml)}`;
  }

  zoom(transform?: d3.ZoomTransform): void {
    if (this.scaleMinimap === null) {
      // Scene is not ready yet.
      return;
    }
    // Update the new translate and scale params, only if specified.
    if (transform) {
      this.translate = [transform.x, transform.y];
      this.scaleMain = transform.k;
    }
    // Update the location of the viewpoint rectangle.
    let svgRect = this.svg.getBoundingClientRect();
    let $viewpoint = d3.select(this.viewpoint);
    this.viewpointCoord.x = (-this.translate[0] * this.scaleMinimap) / this.scaleMain;
    this.viewpointCoord.y = (-this.translate[1] * this.scaleMinimap) / this.scaleMain;
    let viewpointWidth = (svgRect.width * this.scaleMinimap) / this.scaleMain;
    let viewpointHeight = (svgRect.height * this.scaleMinimap) / this.scaleMain;
    $viewpoint
      .attr('x', this.viewpointCoord.x)
      .attr('y', this.viewpointCoord.y)
      .attr('width', viewpointWidth)
      .attr('height', viewpointHeight);
    // Show/hide the minimap depending on the viewpoint area as fraction of the
    // whole minimap.
    let mapWidth = this.minimapSize.width / 2; // 前面乘了这里要除回来
    let mapHeight = this.minimapSize.height;
    let x = this.viewpointCoord.x;
    let y = this.viewpointCoord.y;
    let w = Math.min(Math.max(0, x + viewpointWidth), mapWidth) - Math.min(Math.max(0, x), mapWidth);
    let h = Math.min(Math.max(0, y + viewpointHeight), mapHeight) - Math.min(Math.max(0, y), mapHeight);
    let fracIntersect = (w * h) / (mapWidth * mapHeight);
    if (fracIntersect < FRAC_VIEWPOINT_AREA) {
      this.minimap.classList.remove('hidden');
    } else {
      this.minimap.classList.add('hidden');
    }
  }

  private updateViewpoint(): void {
    // Update the coordinates of the viewpoint rectangle.
    d3.select(this.viewpoint).attr('x', this.viewpointCoord.x).attr('y', this.viewpointCoord.y);
    // Update the translation vector of the main svg to reflect the
    // new viewpoint.
    let mainX = (-this.viewpointCoord.x * this.scaleMain) / this.scaleMinimap;
    let mainY = (-this.viewpointCoord.y * this.scaleMain) / this.scaleMinimap;
    d3.select(this.svg).call(this.mainZoom.transform, d3.zoomIdentity.translate(mainX, mainY).scale(this.scaleMain));
  }
}
