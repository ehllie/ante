%{
#include <stdlib.h>
#include <stdio.h>
#include <tokens.h>
#include <ptree.h>

#ifndef YYSTYPE
#define YYSTYPE Node*
#endif

/* This has no effect when generating a c++ parser */
/* Setting verbose for a c++ parser requires %error-verbose, set in the next section */
#define YYERROR_VERBOSE

#include "yyparser.h"

/* Defined in lexer.cpp */
extern int yylex(yy::parser::semantic_type*, yy::location*);

/*namespace ante{
    extern void error(string& msg, const char *fileName, unsigned int row, unsigned int col);
}*/

void yyerror(const char *msg);

%}

%locations
%error-verbose

%token Ident UserType

/* types */
%token I8 I16 I32 I64 
%token U8 U16 U32 U64
%token Isz Usz F16 F32 F64
%token C8 C32 Bool Void

/* operators */
%token Eq NotEq AddEq SubEq MulEq DivEq GrtrEq LesrEq
%token Or And Range Returns

/* literals */
%token True False
%token IntLit FltLit StrLit

/* keywords */
%token Return
%token If Then Elif Else
%token For While Do In
%token Continue Break
%token Import Let Var Match With
%token Data Enum Fun Ext

/* modifiers */
%token Pub Pri Pro Raw
%token Const Noinit Pathogen

/* other */
%token Where Infect Cleanse Ct

/* whitespace */
%token Newline Indent Unindent


/*
    Now to manually fix all shift/reduce conflicts
*/

/*
    Fake precedence rule to allow for a lower precedence
    than Ident in decl context
*/
%nonassoc LOW

%left Ident
%left Import Return

%left ';' Newline
%left MED

%left Let In
%left ','
%left '=' AddEq SubEq MulEq DivEq

%left Or
%left And     
%left Eq  NotEq GrtrEq LesrEq '<' '>'

%left Range
%left '+' '-'
%left '*' '/' '%'

%left '.'
%left '@' '&'

/* 
    Being below HIGH, this ensures parenthetical expressions will be parsed
    as just order-of operations parenthesis, instead of a single-value tuple.
*/
%nonassoc ')'

%nonassoc '(' '[' Indent Unindent
%nonassoc HIGH

/*
    Expect 4 shift/reduce warnings, all from type casting.  Using a glr-parser
    resolves this ambiguity.
*/
%glr-parser
%expect 4
%start top_level_stmt_list
%%

top_level_stmt_list:  maybe_newline stmt_list maybe_newline
                   ;


stmt_list: stmt_list Newline expr  %prec MED {$$ = setNext($1, $3);}
         | expr                    %prec MED {$$ = setRoot($1);}
         ;


maybe_newline: Newline  %prec Newline
             | %empty   %prec LOW
             ;

/*
 * Statements that will never end with a newline token.
 * Usually statements that require blocks, such as function declarations.
 */

/*
stmt: fn_decl       Newline
    | data_decl     Newline
    | enum_decl     Newline
    | do_while_loop Newline
    | for_loop      Newline
    | var_assign    Newline
    | ret_stmt      Newline
    | extension     Newline
    | expr          Newline
    | import_stmt   Newline
    ;

stmt_no_nl: fn_decl      
          | data_decl    
          | enum_decl    
          | do_while_loop
          | for_loop     
          | var_assign   
          | ret_stmt     
          | extension
          | expr
          | import_stmt
          ;
*/

import_stmt: Import expr {$$ = mkImportNode(@$, $2);}


ident: Ident {$$ = (Node*)lextxt;}
     ;

usertype: UserType {$$ = (Node*)lextxt;}
        ;

intlit: IntLit {$$ = mkIntLitNode(@$, lextxt);}
      ;

fltlit: FltLit {$$ = mkFltLitNode(@$, lextxt);}
      ;

strlit: StrLit {$$ = mkStrLitNode(@$, lextxt);}
      ;

lit_type: I8                        {$$ = mkTypeNode(@$, TT_I8,  (char*)"");}
        | I16                       {$$ = mkTypeNode(@$, TT_I16, (char*)"");}
        | I32                       {$$ = mkTypeNode(@$, TT_I32, (char*)"");}
        | I64                       {$$ = mkTypeNode(@$, TT_I64, (char*)"");}
        | U8                        {$$ = mkTypeNode(@$, TT_U8,  (char*)"");}
        | U16                       {$$ = mkTypeNode(@$, TT_U16, (char*)"");}
        | U32                       {$$ = mkTypeNode(@$, TT_U32, (char*)"");}
        | U64                       {$$ = mkTypeNode(@$, TT_U64, (char*)"");}
        | Isz                       {$$ = mkTypeNode(@$, TT_Isz, (char*)"");}
        | Usz                       {$$ = mkTypeNode(@$, TT_Usz, (char*)"");}
        | F16                       {$$ = mkTypeNode(@$, TT_F16, (char*)"");}
        | F32                       {$$ = mkTypeNode(@$, TT_F32, (char*)"");}
        | F64                       {$$ = mkTypeNode(@$, TT_F64, (char*)"");}
        | C8                        {$$ = mkTypeNode(@$, TT_C8,  (char*)"");}
        | C32                       {$$ = mkTypeNode(@$, TT_C32, (char*)"");}
        | Bool                      {$$ = mkTypeNode(@$, TT_Bool, (char*)"");}
        | Void                      {$$ = mkTypeNode(@$, TT_Void, (char*)"");}
        | usertype  %prec UserType  {$$ = mkTypeNode(@$, TT_Data, (char*)$1);}
        | '\'' ident                {$$ = mkTypeNode(@$, TT_TypeVar, (char*)$1);}
        ;

type: type '*'      %dprec 2             {$$ = mkTypeNode(@$, TT_Ptr,  (char*)"", $1);}
    | type '[' ']'                       {$$ = mkTypeNode(@$, TT_Array,(char*)"", $1);}
    | type '(' type_expr ')'             {$$ = mkTypeNode(@$, TT_Func, (char*)"", $1);}  /* f-ptr w/ params*/
    | type '(' ')'                       {$$ = mkTypeNode(@$, TT_Func, (char*)"", $1);}  /* f-ptr w/out params*/
    | '(' type_expr ')'       %prec MED  {$$ = $2;}
    | lit_type                           {$$ = $1;}
    ;

type_expr_: type_expr_ ',' type {$$ = setNext($1, $3);}
          | type_expr_ '|' type
          | type                {$$ = setRoot($1);}
          ;

type_expr: type_expr_  {Node* tmp = getRoot(); 
                        if(tmp == $1){//singular type, first type in list equals the last
                            $$ = tmp;
                        }else{ //tuple type
                            $$ = mkTypeNode(@$, TT_Tuple, (char*)"", tmp);
                        }
                       }


modifier: Pub      {$$ = mkModNode(@$, Tok_Pub);} 
        | Pri      {$$ = mkModNode(@$, Tok_Pri);}
        | Pro      {$$ = mkModNode(@$, Tok_Pro);}
        | Raw      {$$ = mkModNode(@$, Tok_Raw);}
        | Const    {$$ = mkModNode(@$, Tok_Const);}
        | Noinit   {$$ = mkModNode(@$, Tok_Noinit);}
        | Pathogen {$$ = mkModNode(@$, Tok_Pathogen);}
        ;

modifier_list_: modifier_list_ modifier {$$ = setNext($1, $2);}
              | modifier {$$ = setRoot($1);}
              ;

modifier_list: modifier_list_ {$$ = getRoot();}
             ;


var_decl: maybe_mod_list Var ident '=' expr  {@$ = @3; $$ = mkVarDeclNode(@$, (char*)$3, $1,  0, $5);}
        ;

let_binding: Let modifier_list type_expr ident '=' expr {$$ = mkLetBindingNode(@$, (char*)$4, $2, $3, $6);}
           | Let modifier_list ident '=' expr           {$$ = mkLetBindingNode(@$, (char*)$3, $2, 0,  $5);}
           | Let type_expr ident '=' expr               {$$ = mkLetBindingNode(@$, (char*)$3, 0,  $2, $5);}
           | Let ident '=' expr                         {$$ = mkLetBindingNode(@$, (char*)$2, 0,  0,  $4);}
           ;

/* TODO: change arg1 to require node* instead of char* */
var_assign: ref_val '=' expr    {$$ = mkVarAssignNode(@$, $1, $3);}
          | ref_val AddEq expr  {$$ = mkVarAssignNode(@$, $1, mkBinOpNode(@$, '+', mkUnOpNode(@$, '@', $1), $3), false);}
          | ref_val SubEq expr  {$$ = mkVarAssignNode(@$, $1, mkBinOpNode(@$, '-', mkUnOpNode(@$, '@', $1), $3), false);}
          | ref_val MulEq expr  {$$ = mkVarAssignNode(@$, $1, mkBinOpNode(@$, '*', mkUnOpNode(@$, '@', $1), $3), false);}
          | ref_val DivEq expr  {$$ = mkVarAssignNode(@$, $1, mkBinOpNode(@$, '/', mkUnOpNode(@$, '@', $1), $3), false);}
          ;


usertype_list: usertype_list ',' usertype {$$ = setNext($1, $3);}
             | usertype {$$ = setRoot($1);}
             ;

generic: '<' usertype_list '>' {$$ = getRoot();}
       ;

data_decl: modifier_list Data usertype type_decl_block         {$$ = mkDataDeclNode(@$, (char*)$3, $4);}
         | modifier_list Data usertype generic type_decl_block {$$ = mkDataDeclNode(@$, (char*)$3, $5);}
         | Data usertype type_decl_block                       {$$ = mkDataDeclNode(@$, (char*)$2, $3);}
         | Data usertype generic type_decl_block               {$$ = mkDataDeclNode(@$, (char*)$2, $4);}
         ;

type_decl: type_expr ident {$$ = mkNamedValNode(@$, mkVarNode(@$, (char*)$2), $1);}
         | type_expr       {$$ = mkNamedValNode(@$, 0, $1);}
         | enum_decl
         ;

type_decl_list: type_decl_list Newline type_decl  {$$ = setNext($1, $3);}
              | type_decl                         {$$ = setRoot($1);}
              ;

type_decl_block: Indent type_decl_list Unindent  {$$ = getRoot();}
               ;


/* Specifying an enum member's value */
val_init_list: val_init_list Newline usertype
             | val_init_list Newline usertype '=' expr
             | usertype '=' expr
             | usertype
             ;

enum_block: Indent val_init_list Unindent
          ;

enum_decl: modifier_list Enum usertype enum_block  {$$ = NULL;}
         | Enum usertype enum_block                {$$ = NULL;}
         | modifier_list Enum enum_block           {$$ = NULL;}
         | Enum enum_block                         {$$ = NULL;}
         ;


block: Indent nl_expr Unindent {$$ = $2;}
     ;



raw_ident_list: raw_ident_list ident  {$$ = setNext($1, mkVarNode(@$, (char*)$2));}
              | ident             {$$ = setRoot(mkVarNode(@$, (char*)$1));}
              ;

ident_list: raw_ident_list {$$ = getRoot();}


/* 
 * In case of multiple parameters declared with a single type, eg i32 a b c
 * The next parameter should be set to the first in the list, (the one returned by getRoot()),
 * but the variable returned must be the last in the last, in this case $4
 */


_params: _params ',' type_expr ident_list {$$ = setNext($1, mkNamedValNode(@$, $4, $3));}
      | type_expr ident_list            {$$ = setRoot(mkNamedValNode(@$, $2, $1));}
      ;

                          /* varargs function .. (Range) followed by . */
params: _params ',' Range '.' {setNext($1, mkNamedValNode(@$, mkVarNode(@$, (char*)""), 0)); $$ = getRoot();}
      | _params               {$$ = getRoot();}
      ;

maybe_mod_list: modifier_list  {$$ = $1;}
              | %empty         {$$ = 0;}
              ;

function: fn_def   {$$ = $1;}
        | fn_decl  {$$ = $1;}
        ;

fn_def: maybe_mod_list Fun ident ':' params Returns type_expr block  {$$ = mkFuncDeclNode(@$, /*fn_name*/(char*)$3, /*mods*/$1, /*ret_ty*/$7, /*params*/$5, /*body*/$8);}
      | maybe_mod_list Fun params Returns type_expr block            {$$ = mkFuncDeclNode(@$, /*fn_name*/(char*)0,  /*mods*/$1, /*ret_ty*/$5, /*params*/$3, /*body*/$6);}
      | maybe_mod_list Fun ident ':' Returns type_expr block         {$$ = mkFuncDeclNode(@$, /*fn_name*/(char*)$3, /*mods*/$1, /*ret_ty*/$6, /*params*/0,  /*body*/$7);}
      | maybe_mod_list Fun Returns type_expr block                   {$$ = mkFuncDeclNode(@$, /*fn_name*/(char*)0,  /*mods*/$1, /*ret_ty*/$4, /*params*/0,  /*body*/$5);}
      | maybe_mod_list Fun ident ':' params block                    {$$ = mkFuncDeclNode(@$, /*fn_name*/(char*)$3, /*mods*/$1, /*ret_ty*/0,  /*params*/$5, /*body*/$6);}
      | maybe_mod_list Fun params block                              {$$ = mkFuncDeclNode(@$, /*fn_name*/(char*)0,  /*mods*/$1, /*ret_ty*/0,  /*params*/$3, /*body*/$4);}
      | maybe_mod_list Fun ident ':' block                           {$$ = mkFuncDeclNode(@$, /*fn_name*/(char*)$3, /*mods*/$1, /*ret_ty*/0,  /*params*/0,  /*body*/$5);}
      | maybe_mod_list Fun block                                     {$$ = mkFuncDeclNode(@$, /*fn_name*/(char*)0,  /*mods*/$1, /*ret_ty*/0,  /*params*/0,  /*body*/$3);}
      ;

fn_decl: maybe_mod_list Fun ident ':' params Returns type_expr ';'       {$$ = mkFuncDeclNode(@$, /*fn_name*/(char*)$3, /*mods*/$1, /*ret_ty*/$7, /*params*/$5, /*body*/0);}
       | maybe_mod_list Fun ident ':' Returns type_expr        ';'       {$$ = mkFuncDeclNode(@$, /*fn_name*/(char*)$3, /*mods*/$1, /*ret_ty*/$6, /*params*/0,  /*body*/0);}
       | maybe_mod_list Fun ident ':' params                   ';'       {$$ = mkFuncDeclNode(@$, /*fn_name*/(char*)$3, /*mods*/$1, /*ret_ty*/0,  /*params*/$5, /*body*/0);}
       | maybe_mod_list Fun ident ':'                          ';'       {$$ = mkFuncDeclNode(@$, /*fn_name*/(char*)$3, /*mods*/$1, /*ret_ty*/0,  /*params*/0,  /*body*/0);}
       ;





fn_call: ident tuple {$$ = mkFuncCallNode(@$, (char*)$1, $2);}
       ;


ret_stmt: Return expr {$$ = mkRetNode(@$, $2);}
        ;


extension: Ext type_expr Indent fn_list Unindent {$$ = mkExtNode(@$, $2, $4);}
         ;


fn_list: fn_list_ {$$ = getRoot();}

fn_list_: fn_list_ function maybe_newline  {$$ = setNext($1, $2);} 
        | function maybe_newline           {$$ = setRoot($1);}
        ;


if_stmt: If expr Then expr maybe_newline Else expr  %prec LOW {$$ = mkExprIfNode(@$, $2, $4, $7);}
       ;

while_loop: While expr Do expr  %prec LOW {$$ = mkWhileNode(@$, $2, $4);}
          ;

/*
do_while_loop: Do While expr Do expr {$$ = NULL;}
             ;

for_loop: For ident In expr expr {$$ = NULL;}
        ;
*/

var: ident  %prec Ident {$$ = mkVarNode(@$, (char*)$1);}
   ;


ref_val: '&' ref_val            %prec '&'  {$$ = mkUnOpNode(@$, '&', $2);}
       | '@' ref_val            %prec '@'  {$$ = mkUnOpNode(@$, '@', $2);}
       | ident '[' nl_expr ']'             {$$ = mkBinOpNode(@$, '[', mkRefVarNode(@$, (char*)$1), $3);}
       | ident  %prec Ident                {$$ = mkRefVarNode(@$, (char*)$1);}
       ;


val: fn_call                 {$$ = $1;}
   | '(' nl_expr ')'         {$$ = $2;}
   | tuple                   {$$ = $1;}
   | array                   {$$ = $1;}
   | unary_op                {$$ = $1;}
   | var                     {$$ = $1;}
   | intlit                  {$$ = $1;}
   | fltlit                  {$$ = $1;}
   | strlit                  {$$ = $1;}
   | True                    {$$ = mkBoolLitNode(@$, 1);}
   | False                   {$$ = mkBoolLitNode(@$, 0);}
   | let_binding             {$$ = $1;}
   | var_decl                {$$ = $1;}
   | var_assign              {$$ = $1;}
   | if_stmt                 {$$ = $1;}
   | while_loop              {$$ = $1;}
   | function                {$$ = $1;}
   | data_decl               {$$ = $1;}
   | extension               {$$ = $1;}
   | ret_stmt                {$$ = $1;}
   | import_stmt             {$$ = $1;}
   ;

tuple: '(' expr_list ')' {$$ = mkTupleNode(@$, $2);}
     | '(' ')'           {$$ = mkTupleNode(@$, 0);}
     ;

array: '[' expr_list ']' {$$ = mkArrayNode(@$, $2);}
     | '[' ']'           {$$ = mkArrayNode(@$, 0);}
     ;


unary_op: '@' val                 {$$ = mkUnOpNode(@$, '@', $2);}
        | '&' val                 {$$ = mkUnOpNode(@$, '&', $2);}
        | '-' val                 {$$ = mkUnOpNode(@$, '-', $2);}
        | type_expr val           {$$ = mkTypeCastNode(@$, $1, $2);}
        ;

expr: basic_expr  %prec LOW {$$ = $1;}
    | block                 {$$ = $1;}
    ;

basic_expr: basic_expr '+' basic_expr              {$$ = mkBinOpNode(@$, '+', $1, $3);}
          | basic_expr '-' basic_expr              {$$ = mkBinOpNode(@$, '-', $1, $3);}
          | basic_expr '*' basic_expr              {$$ = mkBinOpNode(@$, '*', $1, $3);}
          | basic_expr '/' basic_expr              {$$ = mkBinOpNode(@$, '/', $1, $3);}
          | basic_expr '%' basic_expr              {$$ = mkBinOpNode(@$, '%', $1, $3);}
          | basic_expr '<' basic_expr              {$$ = mkBinOpNode(@$, '<', $1, $3);}
          | basic_expr '>' basic_expr              {$$ = mkBinOpNode(@$, '>', $1, $3);}
          | basic_expr '.' var                     {$$ = mkBinOpNode(@$, '.', $1, $3);}
          | type_expr  '.' var                     {$$ = mkBinOpNode(@$, '.', $1, $3);}
          | basic_expr ';' basic_expr              {$$ = mkBinOpNode(@$, ';', $1, $3);}
//        | basic_expr Newline basic_expr          {$$ = mkBinOpNode(@$, ';', $1, $3);}
          | basic_expr '[' nl_expr ']'             {$$ = mkBinOpNode(@$, '[', $1, $3);}
          | basic_expr Eq basic_expr               {$$ = mkBinOpNode(@$, Tok_Eq, $1, $3);}
          | basic_expr NotEq basic_expr            {$$ = mkBinOpNode(@$, Tok_NotEq, $1, $3);}
          | basic_expr GrtrEq basic_expr           {$$ = mkBinOpNode(@$, Tok_GrtrEq, $1, $3);}
          | basic_expr LesrEq basic_expr           {$$ = mkBinOpNode(@$, Tok_LesrEq, $1, $3);}
          | basic_expr Or basic_expr               {$$ = mkBinOpNode(@$, Tok_Or, $1, $3);}
          | basic_expr And basic_expr              {$$ = mkBinOpNode(@$, Tok_And, $1, $3);}
          | basic_expr Range basic_expr            {$$ = mkBinOpNode(@$, Tok_Range, $1, $3);}
          | basic_expr tuple                       {$$ = mkBinOpNode(@$, '(', $1, $2);}
          | val                                    {$$ = $1;}
          ;


/* nl_expr is used in expression blocks and can span multiple lines */
expr_list: expr_list_p {$$ = getRoot();}
         ;


expr_list_p: expr_list_p ',' maybe_newline nl_expr  %prec ',' {$$ = setNext($1, $4);}
           | nl_expr                                %prec LOW {$$ = setRoot($1);}
           ;


nl_expr: nl_expr '+' maybe_newline nl_expr                {$$ = mkBinOpNode(@$, '+', $1, $4);}
       | nl_expr '-' maybe_newline nl_expr                {$$ = mkBinOpNode(@$, '-', $1, $4);}
       | nl_expr '*' maybe_newline nl_expr                {$$ = mkBinOpNode(@$, '*', $1, $4);}
       | nl_expr '/' maybe_newline nl_expr                {$$ = mkBinOpNode(@$, '/', $1, $4);}
       | nl_expr '%' maybe_newline nl_expr                {$$ = mkBinOpNode(@$, '%', $1, $4);}
       | nl_expr '<' maybe_newline nl_expr                {$$ = mkBinOpNode(@$, '<', $1, $4);}
       | nl_expr '>' maybe_newline nl_expr                {$$ = mkBinOpNode(@$, '>', $1, $4);}
       | nl_expr '.' maybe_newline var                    {$$ = mkBinOpNode(@$, '.', $1, $4);}
       | type_expr '.' maybe_newline var                  {$$ = mkBinOpNode(@$, '.', $1, $4);}
       | nl_expr ';' maybe_newline nl_expr                {$$ = mkBinOpNode(@$, ';', $1, $4);}
       | nl_expr Newline nl_expr                          {$$ = mkBinOpNode(@$, ';', $1, $3);}
       | nl_expr '[' nl_expr ']'                          {$$ = mkBinOpNode(@$, '[', $1, $3);}
       | nl_expr Eq maybe_newline  nl_expr                {$$ = mkBinOpNode(@$, Tok_Eq, $1, $4);}
       | nl_expr NotEq maybe_newline nl_expr              {$$ = mkBinOpNode(@$, Tok_NotEq, $1, $4);}
       | nl_expr GrtrEq maybe_newline nl_expr             {$$ = mkBinOpNode(@$, Tok_GrtrEq, $1, $4);}
       | nl_expr LesrEq maybe_newline nl_expr             {$$ = mkBinOpNode(@$, Tok_LesrEq, $1, $4);}
       | nl_expr Or maybe_newline nl_expr                 {$$ = mkBinOpNode(@$, Tok_Or, $1, $4);}
       | nl_expr And maybe_newline nl_expr                {$$ = mkBinOpNode(@$, Tok_And, $1, $4);}
       | nl_expr tuple                                    {$$ = mkBinOpNode(@$, '(', $1, $2);}
       | val maybe_newline                                {$$ = $1;}
       | block maybe_newline                              {$$ = $1;}
       ;

%%

/* location parser error */
void yy::parser::error(const location& loc, const string& msg){
    location l = loc;
    ante::error(msg.c_str(), l);
} 

/*
void yy::parser::error(const string& msg){
    ante::error(msg.c_str(), yylexer->fileName, yylexer->getRow(), yylexer->getCol());
}*/
