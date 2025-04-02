import { Component, Input } from '@angular/core';
import { environment } from '../../environments/environment.development';
import { TranslationService } from '../translation.service';

@Component({
  selector: 'app-intelligent-text-area',
  imports: [],
  templateUrl: './intelligent-text-area.component.html',
  styleUrl: './intelligent-text-area.component.css',
  standalone: true
})
export class IntelligentTextAreaComponent {

  sentence: string = ""
  translationService: TranslationService

  constructor(translationService: TranslationService){
    this.translationService = translationService
  }


  onValueChange(value: string){
    this.sentence = value
    let translationObservable = this.translationService.getTranslation(this.sentence)
    translationObservable.subscribe(translation => {
      console.log(translation)
    })

    
  }

  


}
