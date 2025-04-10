import { Component, Input } from '@angular/core';
import { environment } from '../../environments/environment.development';
import { TranslationService } from '../translation.service';
import { time } from 'node:console';
import { ProofReadService } from '../proof-read.service';
import { MatTooltip } from '@angular/material/tooltip'
import { CommonModule } from '@angular/common';
import { Token } from '../interfaces/token';

@Component({
  selector: 'app-intelligent-text-area',
  imports: [ MatTooltip, CommonModule],
  templateUrl: './intelligent-text-area.component.html',
  styleUrl: './intelligent-text-area.component.css',
  standalone: true
})
export class IntelligentTextAreaComponent {

  sentence: string = "";
  translation: string = "";
  tokens: Array<Token> = [ {
            "position": 1,
            "probability": 3.310036117909476e-05,
            "score": -10.315963745117188,
            "token": "ラーメン",
            "top_alternatives": [
                [
                    "それ",
                    0.4243096113204956
                ],
                [
                    "何",
                    0.10090238600969315
                ],
                [
                    "これ",
                    0.07443574070930481
                ],
                [
                    "だ",
                    0.047408681362867355
                ],
                [
                    "なに",
                    0.028508907184004784
                ]
            ]
        },
        {
            "position": 2,
            "probability": 0.05132686346769333,
            "score": -2.969541072845459,
            "token": "が",
            "top_alternatives": [
                [
                    "は",
                    0.4705790579319
                ],
                [
                    "って",
                    0.05779007077217102
                ],
                [
                    "も",
                    0.05707067623734474
                ],
                [
                    "を",
                    0.05408431589603424
                ],
                [
                    "が",
                    0.05132686346769333
                ]
            ]
        },
        {
            "position": 3,
            "probability": 0.010136582888662815,
            "score": -4.591604232788086,
            "token": "いか",
            "top_alternatives": [
                [
                    "な",
                    0.4356398582458496
                ],
                [
                    "す",
                    0.08196546882390976
                ],
                [
                    "騒",
                    0.045646414160728455
                ],
                [
                    "え",
                    0.032859668135643005
                ],
                [
                    "親",
                    0.03268904238939285
                ]
            ]
        },
        {
            "position": 4,
            "probability": 0.00018367623852100223,
            "score": -8.602334976196289,
            "token": "##が",
            "top_alternatives": [
                [
                    "ん",
                    0.8279987573623657
                ],
                [
                    "##ん",
                    0.11626699566841125
                ],
                [
                    "ない",
                    0.014063217677175999
                ],
                [
                    "##い",
                    0.005999579094350338
                ],
                [
                    "へん",
                    0.004681977443397045
                ]
            ]
        },
        {
            "position": 5,
            "probability": 0.030017320066690445,
            "score": -3.5059807300567627,
            "token": "だ",
            "top_alternatives": [
                [
                    "です",
                    0.749268114566803
                ],
                [
                    "の",
                    0.08240805566310883
                ],
                [
                    "でしょう",
                    0.060605376958847046
                ],
                [
                    "だ",
                    0.030017320066690445
                ],
                [
                    "だろう",
                    0.01859746314585209
                ]
            ]
        },
        {
            "position": 6,
            "probability": 0.002912253839895129,
            "score": -5.838828086853027,
            "token": "か",
            "top_alternatives": [
                [
                    "から",
                    0.2008424550294876
                ],
                [
                    "よ",
                    0.1826126128435135
                ],
                [
                    "ね",
                    0.07463380694389343
                ],
                [
                    "な",
                    0.048505984246730804
                ],
                [
                    "ねぇ",
                    0.04515989124774933
                ]
            ]
        },
        {
            "position": 7,
            "probability": 0.6339679956436157,
            "score": -0.455756813287735,
            "token": "。",
            "top_alternatives": [
                [
                    "。",
                    0.6339679956436157
                ],
                [
                    "?",
                    0.29795828461647034
                ],
                [
                    "!",
                    0.019737396389245987
                ],
                [
                    "な",
                    0.003274032846093178
                ],
                [
                    "も",
                    0.0013857621233910322
                ]
            ]
        }
    ]

  translationService: TranslationService;
  proofReadService: ProofReadService;

  timer: NodeJS.Timeout | null = null;
  delay = 3000;

  constructor(
    translationService: TranslationService,
    proofReadService: ProofReadService,
  ){
    this.translationService = translationService;
    this.proofReadService = proofReadService;
  }


  onValueChange(value: string){
    this.sentence = value;

    //Cancel the outdated requests
    if (this.timer){
      clearTimeout(this.timer);
    }

    //Do not request anything if we have and empty sentence
    if (value.length == 0) {
      this.sentence = "";
      this.translation = "";
      return;
    }

    //Schedule the request
    this.timer = setTimeout(() => this.getAnalysis(this), this.delay);
  }

  getAnalysis(ex: IntelligentTextAreaComponent){
    const translationObservable = ex.translationService.getTranslation(this.sentence)
    const proofReadObservable = ex.proofReadService.getProofRead(this.sentence) 
    ex.timer = null
    ex.translation = ""
    ex.tokens = []

    //Get translation from server
    translationObservable.subscribe(t => {
      if (t && typeof(t) == 'string'){
        ex.translation = t
      }
    })

    // Get tokens with all suggestions
    proofReadObservable.subscribe(p => {
      if (p){
        ex.tokens = p
        console.log(ex.tokens)
      }
    })
  }

formatProbability(probability: number): string {
    return (probability * 100).toString().substring(0, 4) + '%'
}
  
getColor(probability: number): string {
    let color1: string = 'rgb(255, 0, 0)'; // First color (red)
    let color2: string = 'rgb(0, 0, 255)'; // Second color (blue)

    const rgb1 = this.parseRgb(color1);
    const rgb2 = this.parseRgb(color2);
        
    const r = Math.round(rgb1.r + (rgb2.r - rgb1.r) * probability);
    const g = Math.round(rgb1.g + (rgb2.g - rgb1.g) * probability);
    const b = Math.round(rgb1.b + (rgb2.b - rgb1.b) * probability);
        
    return `rgb(${r}, ${g}, ${b})`;

}

 parseRgb(rgb: string): { r: number, g: number, b: number } {
    let result = rgb.match(/\d+/g);
    if (!result) result = ['0', '0', '0']
    return {
      r: parseInt(result[0], 10),
      g: parseInt(result[1], 10),
      b: parseInt(result[2], 10)
    };
  }

}
