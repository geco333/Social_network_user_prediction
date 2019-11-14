import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { HttpClientModule } from '@angular/common/http';

import { DataTablesModule } from 'angular-datatables';

import { AppComponent } from './app.component';
import { InputComponent } from './input/input.component';

@NgModule({
   declarations: [
      AppComponent,
      InputComponent
   ],
   imports: [
      BrowserModule,
      DataTablesModule,
      HttpClientModule
   ],
   providers: [],
   bootstrap: [
      AppComponent
   ]
})
export class AppModule { }
