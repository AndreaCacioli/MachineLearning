import { Injectable } from '@angular/core';
import { environment } from '../environments/environment.development';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { format } from 'path';

@Injectable({
  providedIn: 'root'
})
export class TranslationService {

  private readonly apiUrl : string = environment.apiUrl + "/translation"
  httpClient: HttpClient;

  constructor(httpClient: HttpClient) {
    this.httpClient = httpClient
   }

  getTranslation(sentence: string){
    const formData: FormData = new FormData()
    formData.append("sentence", sentence)
    return this.httpClient.post(this.apiUrl, formData, {responseType: "text"});
  }

}
